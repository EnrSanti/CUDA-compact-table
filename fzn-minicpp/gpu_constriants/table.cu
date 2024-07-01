#include "gpu_constriants/table.cuh"
using namespace std;
using namespace Fca;
using namespace Gpu::Memory;

TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){
    setPriority(CLOW);
    //printf("%%%%%% TableGPU constructor\n");

    int noTuples=tuples.size();
    noVars=vars.size();
    currTableSize=(noTuples/32)+1;
    _noVars_dev=mallocDevice<int>(sizeof(int));
    cudaMemcpyAsync(_noVars_dev, &noVars, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    sm_count = device_prop.multiProcessorCount;
    int cores_per_SM = 128;
    //printf("%%%%%% number of SMs: %d\n",sm_count);
    //printf("%%%%%% warp size: %d\n",32);
    //printf("%%%%%% cores per SM: %d\n",cores_per_SM);
    //printf("%%%%%% support size: %d\n",_supportSize);
    
    
    // Memory allocation
    _currTable_dev = mallocDevice<unsigned int >(sizeof(unsigned int)*currTableSize); 
    _currTable_mask_dev = mallocDevice<unsigned int >(sizeof(unsigned int)*currTableSize); 
    _supports_dev = mallocDevice<unsigned int >(sizeof(unsigned int)*_supportSize*currTableSize);
    _supportSize_dev = mallocDevice<int>(sizeof(int));
    _variablesOffsets_dev = mallocDevice<int>(sizeof(int)*noVars);
    _supportOffsetJmp_dev = mallocDevice<int>(sizeof(int)*(noVars+1));
    _currTable_size_dev=mallocDevice<int>(sizeof(int));
    _s_val_size_dev=mallocDevice<int>(sizeof(int));
    _offset=mallocDevice<int>(sizeof(int));
    _s_val_dev=mallocDevice<int>(sizeof(int)*noVars);
    _vars_dev=mallocDevice<unsigned int>(sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    _currTable_reduction_dev=mallocDevice<unsigned int>(sizeof(unsigned int)*currTableSize*4); //matrix
    _output_dev=mallocDevice<int>(sizeof(int)*(currTableSize/32)+1); //one for each block

    //printf("%%%%%% To store %d values i need %d words in my domains\n",_supportSize*currTableSize,((_supportSize/32)+1));
    
    //on host side we create simpler structures to then copy the data
    _currTable_host = mallocHost<unsigned int>(sizeof(unsigned int)*currTableSize); 
    unsigned int *_supports_host = mallocHost<unsigned int>(sizeof(unsigned int)*_supportSize*currTableSize);
    _vars_host=mallocHost<int>(sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    _outputArray=mallocHost<int>(sizeof(int)*(currTableSize/32)+1); 

    //get the vectors to arrays
    for(int i=0;i<_supportSize;i++){
        for(int j=0; j<currTableSize;j++){
            _supports_host[i*currTableSize+j]=_supports[i]._words[j].value();
        }
        
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

            
    //end of could be done better
    *_currTable_host=_currTable._words.data()->value();
    


    //Memory copy
    //cudaMemcpyAsync(_supports_dev, _supports.data(), sizeof(SparseBitSet)*_supportSize, cudaMemcpyHostToDevice);
    //(_currTable_dev, &_currTable, sizeof(SparseBitSet), cudaMemcpyHostToDevice);
    //printf("%%%%%% copying for supportsize: %d, currTableSize %d\n",_supportSize*currTableSize, currTableSize);
    //printing the supports_hosts
    /*for(int i=0;i<_supportSize*currTableSize;i=i+currTableSize){
        if(i%currTableSize==0)
            printf("%%%%%% [%d] ", i);
        for(int j=0;j<currTableSize;j++){
            printBits(_supports_host[i+j]);
        }
        printf("\n");
    }*/
    cudaMemcpyAsync(_supports_dev, _supports_host, sizeof(unsigned int)*_supportSize*currTableSize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_currTable_dev, _currTable_host, sizeof(unsigned int)*currTableSize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_supportSize_dev, &_supportSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_variablesOffsets_dev, _variablesOffsets.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_supportOffsetJmp_dev, _supportOffsetJmp.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(&_supportOffsetJmp_dev[noVars], &_supportSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_currTable_size_dev, &currTableSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_vars_dev, _vars_host, sizeof(unsigned int)*((_supportSize/32)+1), cudaMemcpyHostToDevice);


 
}
void TableGPU::post(){
    //printf("%%%%%% post GPU\n");
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void TableGPU::propagate(){
    //printf("%%%%%% propagate on GPU\n");
    enfoceGAC();
}

void TableGPU::enfoceGAC(){

    int noBlocks=(currTableSize/32)+1;
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
    //printing the s_val

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
    //for each word of currTable launch a kernel
    
    //printf("%%%%%% currTableSize %d, launching %d blocks\n",currTableSize, (currTableSize/32)+1);
    //print(); 
    for(int i=0;i<currTableSize;i++){
        _currTable_host[i]=_currTable._words[i].value();
    }
    cudaMemcpyAsync(_currTable_dev, _currTable_host, sizeof(unsigned int)*currTableSize, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    updateTableGPU<<<noBlocks,32,32*sizeof(unsigned int)>>>(_supports_dev,_s_val_size_dev,_s_val_dev,_supportSize_dev,_variablesOffsets_dev,_supportOffsetJmp_dev,_currTable_dev,_currTable_size_dev,_vars_dev,_currTable_reduction_dev,_output_dev, _offset, _noVars_dev);

	
    cudaDeviceSynchronize();
    
    //retrieve the output from the device
    cudaMemcpyAsync(_outputArray, _output_dev, sizeof(int)*noBlocks, cudaMemcpyDeviceToHost);


    //performed on host, the number of blocks usually is small (e.g. if we have 1280 rows in the table we have 2 blocks)

    for(int i=0; i<noBlocks; i++){
        if(_outputArray[i]==1){
            output=1;
            break;
        }
    }
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
        /*
        for(int i=0;i<currTableSize;i++){
            printf("%%%%%% [%d] ", i);
            printBits(_currTable._words[i].value());
        }*/
        if(_currTable.isEmpty()){
            failNow();
            //printf("%%%%%% backtrack\n");
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
__global__ void updateTableGPU(unsigned int* _supports_dev,int * _s_val_size_dev, int *_s_val_dev, int *_supportSize_dev, int *_variablesOffsets_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* _currTable_dev_size, unsigned int* _vars_dev, unsigned int* _currTable_reduction_dev, int* output, int *offset, int *varNoDev){


    int thPos = blockIdx.x * blockDim.x + threadIdx.x; //which currTable word we are considering
    int varIndex=0;
    extern __shared__ unsigned int mask[]; //mask (32)
    
    //clear mask TODO MANDATORY

    mask[threadIdx.x]=0;
    
    //for each word in my column in supports
    if(thPos>=*_currTable_dev_size){
        return;
    }

    int varNo=0;
    
    for(int i=0; i<*_s_val_size_dev; i++){
        varIndex=_s_val_dev[i];
        int loops=_supportOffsetJmp_dev[varIndex+1]-_supportOffsetJmp_dev[varIndex];    
        int from=_supportOffsetJmp_dev[varIndex];
        
        //printf("%%%%%% GPU th %d var %d from %d to %d\n",thPos,varIndex,from,from+loops);
        //checking if the var is in s_val
        for(int j=0; j<loops; j++){
            int wordIndex=(from+j)/32; //row
            int maskContains=1<<(31-j-_supportOffsetJmp_dev[varIndex]+wordIndex*32);

            //printf("%%%%%% GPU th %d var %d, accessing word %d, maskContains: %u\n",thPos,varIndex,wordIndex, maskContains);
            //printf("%%%%%% GPU th %d var %d maskContains %u\n",thPos,varIndex,maskContains);
            if(_vars_dev[wordIndex] & maskContains){ //check if val in domain
                //printf("%%%%%% GPU INSIDE th %d var %d contains %d\n",thPos,varIndex,j);
                int off=j*(*_currTable_dev_size)+(_supportOffsetJmp_dev[varIndex]*(*_currTable_dev_size))+threadIdx.x; //1 -> the size of the currTable
                //printf("%%%%%% GPU INSIDE th %d off %d, _currTable_dev_size: %u,_supportOffsetJmp_dev[varIndex]: %d\n",thPos,off,*_currTable_dev_size,_supportOffsetJmp_dev[varIndex]);
                mask[threadIdx.x]=mask[threadIdx.x] | _supports_dev[off];
                //printf("%%%%%% GPU INSIDE th %d mask related to var %d is %u, only the mask %u (accessing %d)\n",thPos,varIndex,mask[threadIdx.x],_supports_dev[off],off);
            }
            
            __syncthreads();
        }
        //printing complete mask
        //printf("%%%%%% GPU th %d complete mask for var %d is %u, table before[%d] %u\n",thPos,varIndex,mask[threadIdx.x],thPos,_currTable_dev[thPos]);
        _currTable_dev[thPos]=mask[threadIdx.x] & _currTable_dev[thPos];
        mask[threadIdx.x]=0;
    }
    
    //printf("%%%%%% GPU th %d currTable[%d] %d\n",thPos,thPos,_currTable_dev[thPos]);
    if(threadIdx.x==0){
        /*
        for(int i=0; i<*_supportSize_dev*2; i++){
            printf("%%%%%% [%d] %u\n",i,_supports_dev[i]);
        }
        //Forall all the vars
        for(int i=0; i<(*_s_val_size_dev)+1; i++){
           int k=_supportOffsetJmp_dev[varIndex];
           printf("%%%%%% GPU th %d var %d offset %d\n",thPos,i,_supportOffsetJmp_dev[i]);
        }
        */
        //printf("%%%%%% GPU kernel over\n");
        for(int i=blockIdx.x*32;i<(blockIdx.x+1)*32;i++){
            if(_currTable_dev[i]!=0){
                output[blockIdx.x]=0;
                return;
            }
        }
        output[blockIdx.x]=1;
        //printf("%%%%%% GPU fail now GPU\n");
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
    printf("%%%%%% --------------------------------------------------------\n");
}