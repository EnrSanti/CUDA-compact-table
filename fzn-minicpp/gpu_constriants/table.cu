#include "gpu_constriants/table.cuh"
using namespace std;
using namespace Fca;
using namespace Gpu::Memory;

TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){
    setPriority(CLOW);
    printf("%%%%%% TableGPU constructor\n");

    int noTuples=tuples.size();
    noVars=vars.size();
    currTableSize=(noTuples/32)+1;

    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    sm_count = device_prop.multiProcessorCount;
    int cores_per_SM = 128;
    printf("%%%%%% number of SMs: %d\n",sm_count);
    printf("%%%%%% warp size: %d\n",32);
    printf("%%%%%% cores per SM: %d\n",cores_per_SM);
    printf("%%%%%% support size: %d\n",_supportSize);
    
    
    // Memory allocation
    _currTable_dev = mallocDevice<unsigned int >(sizeof(unsigned int)*currTableSize); 
    _currTable_mask_dev = mallocDevice<unsigned int >(sizeof(unsigned int)*currTableSize); 
    _supports_dev = mallocDevice<unsigned int >(sizeof(unsigned int)*_supportSize*currTableSize);
    _supportSize_dev = mallocDevice<int>(sizeof(int));
    _variablesOffsets_dev = mallocDevice<int>(sizeof(int)*noVars);
    _supportOffsetJmp_dev = mallocDevice<int>(sizeof(int)*noVars);
    _currTable_size_dev=mallocDevice<int>(sizeof(int));
    _s_val_size_dev=mallocDevice<int>(sizeof(int));
    _offset=mallocDevice<int>(sizeof(int));
    _s_val_dev=mallocDevice<int>(sizeof(int)*noVars);
    _vars_dev=mallocDevice<unsigned int>(sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    _currTable_reduction_dev=mallocDevice<unsigned int>(sizeof(unsigned int)*currTableSize*4); //matrix
    _output_dev=mallocDevice<int>(sizeof(int));
    printf("%%%%%% To store %d values i need %d words in my domains\n",_supportSize*currTableSize,((_supportSize/32)+1));
    
    //on host side we create simpler structures to then copy the data
    _currTable_host = mallocHost<unsigned int>(sizeof(unsigned int)*currTableSize); 
    unsigned int *_supports_host = mallocHost<unsigned int>(sizeof(unsigned int)*_supportSize*currTableSize);
    _vars_host=mallocHost<int>(sizeof(unsigned int)*((_supportSize/32)+1)); //matrix


    //get the vectors to arrays
    for(int i=0;i<_supportSize;i++){
        _supports_host[i*currTableSize]=_supports[i]._words.data()->value();
    }

    for(int i=0;i<((_supportSize/32)+1);i++){
        _vars_host[i]=0;
    }

    //can be done much better but for now it's ok
    for(int i=0;i<noVars;i++){
        vector<int> dom=_vars[i]->dumpDomainToVec();
        for(int j=0;j<dom.size();j++){
            //getting an unsigned int with the 32-dom[j]-_variablesOffsets[i] bit set
            unsigned int mask=1<<31-(dom[j]-_variablesOffsets[i]+_supportOffsetJmp[i]);
            int starting_word=(dom[j]-_variablesOffsets[i]+_supportOffsetJmp[i])/32;
            _vars_host[starting_word]=_vars_host[starting_word]|mask;
            //prinitng bits of _vars_host
        }
    }

    printf("%%%%%% the vars squished together[0] : %u\n",_vars_host[0]);
            
    //end of could be done better

    *_currTable_host=_currTable._words.data()->value();
    

    //printf("%%%%%% check sizes 1: %lu %d \n",_currTable._words.size(),currTableSize);
    //printf("%%%%%% check sizes 2: %lu %d \n",_supportSize*currTableSize,_supports.size()*_currTable._words.size());
    //printing currTable info
    //printf("%%%%%% currTable size: %u\n", _currTable._words.size());
    printf("%%%%%% --------------------------------------------------------\n");
    //Memory copy
    //cudaMemcpyAsync(_supports_dev, _supports.data(), sizeof(SparseBitSet)*_supportSize, cudaMemcpyHostToDevice);
    //(_currTable_dev, &_currTable, sizeof(SparseBitSet), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_supports_dev, _supports_host, sizeof(unsigned int)*_supportSize*currTableSize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_currTable_dev, _currTable_host, sizeof(unsigned int)*currTableSize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_supportSize_dev, &_supportSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_variablesOffsets_dev, _variablesOffsets.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_supportOffsetJmp_dev, _supportOffsetJmp.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_currTable_size_dev, &currTableSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_vars_dev, _vars_host, sizeof(unsigned int)*((_supportSize/32)+1), cudaMemcpyHostToDevice);
    //printf("%%%%%% After mem cpy\n");
    //printGPUdata<<<1,1>>>(_supportSize_dev,_variablesOffsets_dev,_currTable_dev,_supports_dev,_supportOffsetJmp_dev,_currTable_size_dev);
    //cudaDeviceSynchronize();


     //just printint
    
    int k=0;
    int off=0;
    for(int i=0;i<_supportSize;i++){
        if(i==_supportOffsetJmp[k]){
            printf("%%%%%% VAR\n");
            k++;
        }
        for(int j=0;j<currTableSize;j++){
            //we need to unwrap the bits
            printf("%%%%%% [%d] ", _variablesOffsets[k]+i);
            printBits(_supports_host[i*currTableSize+j]);
        }   
    }
}
void TableGPU::post(){
    //printf("%%%%%% post GPU\n");
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void TableGPU::propagate(){
    printf("%%%%%% propagate on GPU\n");
    enfoceGAC();
}

void TableGPU::enfoceGAC(){

    print();    
    cudaMemcpyAsync(_currTable_dev, _currTable_host, sizeof(unsigned int)*currTableSize, cudaMemcpyHostToDevice);
    //reset var_host
    for(int i=0;i<((_supportSize/32)+1);i++){
        _vars_host[i]=0;
    }
    //can be done much better but for now it's ok
    for(int i=0;i<noVars;i++){
        vector<int> dom=_vars[i]->dumpDomainToVec();
        
        for(int j=0;j<dom.size();j++){
            //getting an unsigned int with the 32-dom[j]-_variablesOffsets[i] bit set
            unsigned int mask=1<<31-(dom[j]-_variablesOffsets[i]+_supportOffsetJmp[i]);
            //printing the domain
            printf("%%%%%% redefining var[%d]: %u \n ",i,mask);
            int starting_word=(dom[j]-_variablesOffsets[i]+_supportOffsetJmp[i])/32;
            _vars_host[starting_word]=_vars_host[starting_word]|mask;
        }
    }
    cudaMemcpyAsync(_vars_dev, _vars_host, sizeof(unsigned int)*((_supportSize/32)+1), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
            
    //end of could be done better

    _s_val.clear();
    _s_sup.clear();
    int output=0;
	for (int i = 0; i < _vars.size(); i++){
		//update s_val and the deltas
        if(_vars[i]->changed()){
            _s_val.push_back(i);
        }
        //update s_sup
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
        }
	}
    
    int size=_s_val.size();
    printf("%%%%%% HERE 2 printf sval size: %d, val 0: %d\n",size,_s_val[0]);
    cudaMemcpyAsync(_s_val_size_dev, &size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_s_val_dev, _s_val.data(), sizeof(int)*size, cudaMemcpyHostToDevice);
    int offset=0;
    int min=noVars;
    for(int i=0;i<size;i++){
        if(_s_val[i]<min){
            offset=_s_val[i];
        }
    }
    cudaMemcpyAsync(_offset, &offset, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    //printGPUdata<<<1,1>>>(_supportSize_dev,_variablesOffsets_dev,_currTable_dev,_supports_dev,_supportOffsetJmp_dev,_currTable_size_dev);
    //cudaDeviceSynchronize();
    
	updateTableGPU<<<4,32,130*sizeof(unsigned int)>>>(_supports_dev,_s_val_size_dev,_s_val_dev,_supportSize_dev,_variablesOffsets_dev,_supportOffsetJmp_dev,_currTable_dev,_currTable_size_dev,_vars_dev,_currTable_reduction_dev,_output_dev, _offset);
    
    cudaDeviceSynchronize();
    
    //retrieve the output from the device
    cudaMemcpyAsync(&output, _output_dev, sizeof(int), cudaMemcpyDeviceToHost);
 
    if(output==1){
        failNow();
        printf("%%%%%% fail now\n");
    }else{
        //we retrieve current table
        //getting back the current table
       
        cudaMemcpyAsync(_currTable_host, _currTable_dev, sizeof(unsigned int)*(currTableSize), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        //we need to update the current table

        for(int i=0;i<currTableSize;i++){
            _currTable._mask[i]=_currTable_host[i];
        }
        
        _currTable.intersectWithMask();
        _currTable.clearMask();
        //printing the currTable
        for(int i=0;i<currTableSize;i++){
            printf("%%%%%% [%d] ", i);
            printBits(_currTable._words[i].value());
        }
        //then we reput the mask on the device
        cudaMemcpyAsync(_currTable_dev, _currTable_host, sizeof(unsigned int)*currTableSize, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        if(_currTable.isEmpty()){
            failNow();
            printf("%%%%%% backtrack\n");
        }
    }
    
    
	filterDomains();

    

}
void TableGPU::filterDomains(){
    for(int i=0; i < _s_sup.size(); ++i){
        int index=_s_sup[i];
        //printf("%%%%%% filtering domain for var %d\n",index);
        for (int j = 0; j < _vars[index]->size(); j++){
            if(_vars[index]->contains(j+_vars[index]->initialMin())){ //i.e. a \in dom(x)

                int index_x_a=_supportOffsetJmp[index]+j;
                int indexResidue=_residues[index_x_a].value();

                if((_currTable._words[indexResidue] & _supports[index_x_a]._words[indexResidue] ) == 0x00000000){
                
                    indexResidue=_supports[index_x_a].intersectIndexSparse(_currTable);
                    
                    if(indexResidue!=-1){
                        _residues[index_x_a].setValue(indexResidue); //ok setVal
                    }else{
                        _vars[index]->remove(j+_vars[index]->initialMin());                   
                    }
                  
                }
                
            }
        }
        _vars[index]->dumpInSparseBitSet(_vars[index]->min(),_vars[index]->max(),_lastVarsValues[index]);
    }
}
// 1 th per support row
__global__ void updateTableGPU(unsigned int* _supports_dev,int * _s_val_size_dev, int *_s_val_dev, int *_supportSize_dev, int *_variablesOffsets_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* _currTable_dev_size, unsigned int* _vars_dev, unsigned int* _currTable_reduction_dev, int* output, int *offset){
 

    

    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    int varIndex=-1;
    extern __shared__ unsigned int mask[]; //mask 
    //clear mask TODO MANDATORY
    mask[0]=0;
    
    if(thPos >=* _supportSize_dev){
        return;
    }
     
    
    
    if(thPos==0){
        printf("%%%%%% GPU: size: %d \n",*_s_val_size_dev);
        for(int i=0;i<*_s_val_size_dev;i++){
            printf("%%%%%% GPU: _s_val_dev[%d] = %d, offset jmp%d\n",i,_s_val_dev[i],_supportOffsetJmp_dev[i]);
        }
    }
    //every th checks which var it will be working on
    //TODO add check on the size of _s_val_dev oppure non mettere if(_vars[i]->changed()){ e invece che 3 metti _s_val_dev.size()
    for(int i=0; i<3; i++){ // 3 è var no
        if(thPos >= _supportOffsetJmp_dev[i]){
            varIndex++;
        }else{
            break;
        }
    }
    int found=0;
    for(int i=0; i<*_s_val_size_dev; i++){
        if(varIndex==_s_val_dev[i]){
            found=1;
            break;
        }
    }
    if(found==0){
        return;
    }

    int thPosScaled=thPos-_supportOffsetJmp_dev[*offset];
    printf("%%%%%% GPU: thPos %d, varIndex %d\n",thPos,varIndex);
    //delete printing
    if(thPosScaled==0){
        for(int i=0;i<*_currTable_dev_size;i++){
           
            printf("\n%%%%%% GPU on  currTable: ");
            int num=_currTable_dev[i];
            char str[32] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
            for (int i = 31; i >= 0; i--) {
                str[i] = (num >> i) & 1; 
                printf("%d",str[i]);
            }
            printf("\n%%%%%% \n");
            //end delete
    }

    }
    __syncthreads();

    int index=_s_val_dev[varIndex]; //index tells me which var to access
    
    //check just _vars_dev[index]
    int index_x_a=thPos;
    //check if the bit at thPos of _vars_dev[index] is set to 1
    int word=thPos/32;
    int bit=thPos%32;
    unsigned int mask2=1<<(31-bit);
    //printf("%%%%%% i am thread %d and i am checking the bit %d of word %d of var %d, mask %u, the and value %d\n",thPos,bit,word,index,mask2, _vars_dev[word]&mask2);
    //RESET BASED UPDATE

    //each th deals with one bit of the word (i.e. one var value in the domain)
    if ((_vars_dev[word]&mask2)!=0) {
       // printf("%%%%%% GPU: th %d, var %d, value %d\n",thPos,index,thPos);
        for(int i=0;i<*_currTable_dev_size;i++){
            printf("%%%%%% GPU: mask2[%d] = %u, _supports_dev = %u\n",i,mask2,_supports_dev[index_x_a*(*_currTable_dev_size)+i]);
            atomicOr(&mask[i],_supports_dev[index_x_a*(*_currTable_dev_size)+i]);
            printf("%%%%%% GPU: mask[%d] = %u\n",i,mask[i]);\
        }
    } 
    //the fist th of the block will store the mask in the reduction matrix
    if(threadIdx.x==0 || thPosScaled==0){ //FIX
        printf("%%%%%% HEHE GPU th 0 of block %d, printing mask %u\n",blockIdx.x, mask[0]);
        for(int i=0;i<*_currTable_dev_size;i++){
            _currTable_reduction_dev[blockIdx.x*(*_currTable_dev_size)+i]=mask[i];
        }
    }

    if(thPosScaled==0){
        printf("%%%%%% GPU on  the vars squished together[0] : %u QUI QUI %d\n",_vars_dev[0],mask[0]);
        //delete printing 
        for(int i=0;i<*_currTable_dev_size;i++){
           
            printf("\n%%%%%% GPU on currTable: ");
            int num=_currTable_dev[i];
            char str[32] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
            for (int i = 31; i >= 0; i--) {
                str[i] = (num >> i) & 1; 
                printf("%d",str[i]);
            }
            printf("\n%%%%%% \n");
            //end delete
        }
        for(int i=0;i<*_currTable_dev_size;i++){
            for(int j=0; j<4; j++){
                printf("\n%%%%%% GPU on  _currTable_reduction_dev[%d]: %u", i,_currTable_reduction_dev[j*(*_currTable_dev_size)+i]);
                mask[i]=mask[i] | _currTable_reduction_dev[j*(*_currTable_dev_size)+i];
            }
            _currTable_dev[i] =_currTable_dev[i] & mask[i];
        }
        //print mask 0
        printf("%%%%%% GPU GPU mask[0] HERE: %u\n",mask[0]);
         //delete printing 
        for(int i=0;i<*_currTable_dev_size;i++){
           
            printf("\n%%%%%% GPU currTable: ");
            int num=_currTable_dev[i];
            char str[32] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
            for (int i = 31; i >= 0; i--) {
                str[i] = (num >> i) & 1; 
                printf("%d",str[i]);
            }
            printf("\n%%%%%% \n");
            //end delete
        }
        printf("%%%%%%GPU kernel over");
        int empty=1;
        for(int i=0;i<*_currTable_dev_size;i++){
            if(_currTable_dev[i]!=0){
                *output=0;
                return;
            }
        }
        *output=1;
        printf("%%%%%% GPU fail now GPU\n");
    }
   

}
//utilities
__global__ void printGPUdata(int *_supportSize_dev, int *_variablesOffsets_dev,unsigned int *_currTable_dev,unsigned int *_supports_dev,int * _supportOffsetJmp_dev, int* currTable_size_dev){
    printf("%%%%%% -------------------------- printGPUdata -------------------------- \n");
    printf("%%%%%% threadIdx.x: %d\n",threadIdx.x);
    printf("%%%%%% _supportSize_dev: %d\n",*_supportSize_dev);
    //printing the offsets
    printf("%%%%%% _variablesOffsets_dev: %d \n",_supportOffsetJmp_dev[0]);
    printf("%%%%%% _variablesOffsets_dev: %d \n",_supportOffsetJmp_dev[1]);
    printf("%%%%%% _variablesOffsets_dev: %d \n",_supportOffsetJmp_dev[2]);
    int k=0;
    int off=0;
    for(int i=0;i<*_supportSize_dev;i++){
        if(i==_supportOffsetJmp_dev[k]){
            printf("%%%%%% VAR %d\n",k);
            k++;
        }
        for(int j=0;j<*currTable_size_dev;j++){
            //we need to unwrap the bits
            printf("%%%%%% [%d] ", _variablesOffsets_dev[k]+i);
            printBitsGPU(_supports_dev[i*(*currTable_size_dev)+j]);
        }   
    }
    printf("%%%%%% currTable\n");
    for(int j=0;j<*currTable_size_dev;j++){
        printf("%%%%%% [%d] ", j);
        printBitsGPU(_currTable_dev[j]);
    }
 
}
void printBits(unsigned int num) {
    // Extracting each bit of the int and printing it
    //yes rather weird function, but since we need to print %%%%%
    char str[32] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
    for (int i = 31; i >= 0; i--) {
        str[i] = (num >> i) & 1; 
        printf("%d",str[i]);
    }
    printf("\n%%%%%% \n");
}
__device__ void printBitsGPU(unsigned int num) {
    // Extracting each bit of the int and printing it
    //yes rather weird function, but since we need to print %%%%%
    char str[32] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
    for (int i = 31; i >= 0; i--) {
        str[i] = (num >> i) & 1; 
        printf("%d",str[i]);
    }
    printf("\n%%%%%% \n");
}
void TableGPU::print(){    
    
    printf("%%%%%% ----------------- VARS: -----------------\n\n");
    for (int i = 0; i < _vars.size(); i++){
        printf("%%%%%% Var %d: %d\n",i,_vars[i]->getId());      
        printf("%%%%%% Var %d dump: %d \n",i,_lastVarsValues[i]._words[0].value());
    }
    for (int i = 0; i < _vars.size(); i++){
        //checking the contained values
        for (int j = 0; j < _vars[i]->intialSize(); j++){
            if(_vars[i]->contains(j+_vars[i]->initialMin()))
                printf("%%%%%% Var %d contains? %d: YES\n",i,j+_vars[i]->initialMin());
            else
                printf("%%%%%% Var %d contains? %d: NO\n",i,j+_vars[i]->initialMin());
        }
    }
    printf("%%%%%% ----------------- CURR TABLE: -----------------\n\n");
    for (int i = 0; i < _currTable._words.size(); i++){
        printf("%%%%%% [%d] ", i);
        printBits(_currTable._words[i].value());
    }
}