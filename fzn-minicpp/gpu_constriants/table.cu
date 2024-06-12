#include "gpu_constriants/table.cuh"
using namespace std;
using namespace Fca;
using namespace Gpu::Memory;

TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){
    setPriority(CLOW);
    printf("%%%%%% TableGPU constructor\n");

    int noTuples=tuples.size();
    int noVars=vars.size();
    int currTableSize=(noTuples/32)+1;

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
    _s_val_dev=mallocDevice<int>(sizeof(int)*noVars);
    _vars_dev=mallocDevice<unsigned int>(sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    _currTable_reduction_dev=mallocDevice<unsigned int>(sizeof(unsigned int)*currTableSize*4); //matrix
    _output_dev=mallocDevice<int>(sizeof(int));
    printf("%%%%%% To store %d values i need %d words in my domains\n",_supportSize*currTableSize,((_supportSize/32)+1));
    //on host side
    unsigned int *_currTable_host = mallocHost<unsigned int>(sizeof(unsigned int)*currTableSize); 
    unsigned int *_supports_host = mallocHost<unsigned int>(sizeof(unsigned int)*_supportSize*currTableSize);
    int *_vars_host=mallocHost<int>(sizeof(unsigned int)*((_supportSize/32)+1)); //matrix


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
        }
    }
    //printf("%%%%%% _supportOffsetJmp[%d] %d\n",0,_supportOffsetJmp[0]);
    //printf("%%%%%% _supportOffsetJmp[%d] %d\n",1,_supportOffsetJmp[1]);
    //printf("%%%%%% _supportOffsetJmp[%d] %d\n",1,_supportOffsetJmp[2]);
    //end of could be done better

    *_currTable_host=_currTable._words.data()->value();
    

    

    //just printint
    /*
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
    printf("%%%%%% currTable\n");
    for(int j=0;j<currTableSize;j++){
        printf("%%%%%% [%d] ", j);
        printBits(_currTable_host[j]);
    }
    */
    //end printing



    printf("%%%%%% check sizes 1: %lu %d \n",_currTable._words.size(),currTableSize);
    printf("%%%%%% check sizes 2: %lu %d \n",_supportSize*currTableSize,_supports.size()*_currTable._words.size());
    //printing currTable info
    printf("%%%%%% currTable: %u\n", _currTable._words.size());

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
    printf("%%%%%% After mem cpy\n");
    printGPUdata<<<1,1>>>(_supportSize_dev,_variablesOffsets_dev,_currTable_dev,_supports_dev,_supportOffsetJmp_dev,_currTable_size_dev);
    cudaDeviceSynchronize();
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
    _s_val.clear();
    _s_sup.clear();
    
	for (int i = 0; i < _vars.size(); i++){
		//update s_val and the deltas
        //if(_vars[i]->changed()){
            _s_val.push_back(i);
        //}else{
        //}
		//update s_sup
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
        }
	}
    int size=_s_val.size();
    cudaMemcpyAsync(_s_val_size_dev, &size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_s_val_dev, _s_val.data(), sizeof(int)*size, cudaMemcpyHostToDevice);
	updateTableGPU<<<4,32,130*sizeof(unsigned int)>>>(_supports_dev,_s_val_size_dev,_s_val_dev,_supportSize_dev,_variablesOffsets_dev,_supportOffsetJmp_dev,_currTable_dev,_currTable_size_dev,_vars_dev,_currTable_reduction_dev,_output_dev);
    cudaDeviceSynchronize();
    int output;
    //retrieve the output from the device
    cudaMemcpyAsync(&output, _output_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if(output==1){
        failNow();
        printf("%%%%%% fail now\n");
    }
    //printBits(_currTable._words.data()->value()
	//filterDomains();
}
// 1 th per support row
__global__ void updateTableGPU(unsigned int* _supports_dev,int * _s_val_size_dev, int *_s_val_dev, int *_supportSize_dev, int *_variablesOffsets_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* cur_currTable_dev_size, unsigned int* _vars_dev, unsigned int* _currTable_reduction_dev, int* output){
  
    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ unsigned int mask[]; //mask 
    int index=0;
    
    if(thPos>=*_supportSize_dev){
        return;
    }
    
    
    //clear mask TODO

    int varIndex=0;
    for(int i=0; i<3; i++){
        if(thPos>=_supportOffsetJmp_dev[i]){
            varIndex++;
        }else{
            break;
        }
    }
    
    __syncthreads();
    /*if(varIndex>=*_s_val_size_dev){
        return;
    }*/
    index=_s_val_dev[varIndex]; //index tells me which var to access
    
    //if(index==-1)
    //    return;

    //reset based update
    //printf("%%%%%% reset based update \n");
 
    
    //check just _vars_dev[index]
    int index_x_a=thPos;
    printf("%%%%%% i am th: %d, related to var %d, looking at dom %d\n",thPos,varIndex,thPos);
    //check if the bit at thPos of _vars_dev[index] is set to 1
    int word=thPos/32;
    int bit=thPos%32;
    unsigned int mask2=1<<31-bit;
    printf("%%%%%% am i 0? %d, %d",_vars_dev[word]&mask2,*cur_currTable_dev_size);
    if ((_vars_dev[word]&mask2)!=0) {
        for(int i=0;i<*cur_currTable_dev_size;i++){
            atomicOr(&mask[i],_supports_dev[index_x_a/32+i]);
            printf("%%%%%% accessing [%d] word of supports %d ", index_x_a/32+i, _supports_dev[index_x_a/32+i]);
        }
    } 
    if(threadIdx.x==0){
        printf("%%%%%% th 0 of block %d\n",blockIdx.x);
        for(int i=0;i<*cur_currTable_dev_size;i++){
            _currTable_reduction_dev[blockIdx.x*(*cur_currTable_dev_size)+i]=mask[i];
        }
    }

    if(thPos==0){
        for(int i=0;i<*cur_currTable_dev_size;i++){
            for(int j=0; j<4; j++){
                mask[i]=mask[i] | _currTable_reduction_dev[j*(*cur_currTable_dev_size)+i];
            }
        }
        //_currTable.intersectWithMask();
        
        for(int i=0;i<*cur_currTable_dev_size;i++){
            _currTable_dev[i]=_currTable_dev[i] & mask[i];

        }
        int empty=1;
        for(int i=0;i<*cur_currTable_dev_size;i++){
            if(_currTable_dev[i]!=0){
                *output=0;
                return;
            }
        }
        *output=1;
        printf("%%%%%% fail now GPU\n");
        //set __device__ var TODO
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
            printf("%%%%%% VAR\n");
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