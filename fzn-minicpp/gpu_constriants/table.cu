#include "gpu_constriants/table.cuh"

TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){
    setPriority(CLOW);
    //printf("%%%%%% TableGPU constructor\n");

    int noTuples=tuples.size();
    noVars=vars.size();
    
    currTableSize=(noTuples/32)+1; 
    

    // Memory allocation
    _noVars_dev=mallocDevice<int>(sizeof(int));
    _currTable_dev = mallocDevice<unsigned int >(sizeof(unsigned int)*currTableSize); 
    _currTable_mask_dev = mallocDevice<unsigned int >(sizeof(unsigned int)*currTableSize); 
    _supports_dev = mallocDevice<unsigned int>(sizeof(unsigned int)*_supportSize*currTableSize);
    _supportSize_dev = mallocDevice<int>(sizeof(int));
    _variablesOffsets_dev = mallocDevice<int>(sizeof(int)*noVars);
    _supportOffsetJmp_dev = mallocDevice<int>(sizeof(int)*(noVars+1));
    _currTable_size_dev=mallocDevice<int>(sizeof(int));
    _svSize_sval_dev=mallocDevice<int>(sizeof(int)*(noVars+1));
    _vars_dev=mallocDevice<unsigned int>(sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    _output_dev=mallocDevice<int>(sizeof(int)*(currTableSize/32)+1); //one for each block
    offset_dev=mallocDevice<int>(sizeof(int)*noStreams);
    workerOffestAndLimit_dev=mallocDevice<int>(sizeof(int)*16*noVars);
    
    
    

    //on host side we create simpler structures to then copy the data
    int * offset_host;
    cudaMallocHost((void**)&_currTable_host, sizeof(unsigned int)*currTableSize);
    cudaMallocHost((void**)&_vars_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&_outputArray, sizeof(int)*(currTableSize/32)+1);
    cudaMallocHost((void**)&_svSize_sval_host,sizeof(int)*(noVars+1));
    cudaMallocHost((void**)&offset_host,sizeof(int)*noStreams);
    cudaMallocHost((void**)&stream_buffer,sizeof(int)*noStreams);

    cudaMallocHost((void**)&CTsizes_host,sizeof(int)*noStreams);
    cudaMallocHost((void**)&ss32_host,sizeof(int)*noStreams);
    cudaMallocHost((void**)&noBlocks_host,sizeof(int)*noStreams);
    cudaMallocHost((void**)&workerOffestAndLimit_host,sizeof(int)*16*noVars);


    streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*noStreams);
    //printf("%%%%%% creating %d streams\n",noStreams);



    for(int i=0;i<((_supportSize/32)+1);i++){
        _vars_host[i]=0xffffffff;
    }

    for(int i=0;i<noStreams;i++){
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess){
            printf("%%%%%% Stream err: %s\n", cudaGetErrorString(err));
            fflush(stdout);
            runtime_error("Error creating stream");
        }
    }


    cudaMemcpyAsync(_noVars_dev, &noVars, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    *_currTable_host=_currTable._words.data()->value();
    
    //Memory copy

    cudaMemcpyAsync(_supports_dev, _supports, sizeof(unsigned int)*_supportSize*currTableSize, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_currTable_dev, _currTable_host, sizeof(unsigned int)*currTableSize, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supportSize_dev, &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[1]);
    cudaMemcpyAsync(_variablesOffsets_dev, _variablesOffsets.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[1]);
    cudaMemcpyAsync(_supportOffsetJmp_dev, _supportOffsetJmp.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[2]);
    cudaMemcpyAsync(&_supportOffsetJmp_dev[noVars], &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[3]);
    cudaMemcpyAsync(_currTable_size_dev, &currTableSize, sizeof(int), cudaMemcpyHostToDevice,streams[3]);


    //compute once and transfer the offsets for the streams:
    noBlocks=(currTableSize/32)+1;
    divideInStrems(noBlocks,offset_host);
    stream_buffer[0]=0;
    for(int i=1; i<noStreams; i++){
        if(offset_host[i]>0){
            stream_buffer[i]=stream_buffer[i-1]+offset_host[i-1];
        }
    }
    

    for(int i=0;i<noVars-1;i++){
        varOffsetLimit(_supportOffsetJmp[i+1]-_supportOffsetJmp[i],workerOffestAndLimit_host+(i*16));
    }
    varOffsetLimit(_supportSize-_supportOffsetJmp[noVars-1],workerOffestAndLimit_host+((noVars-1)*16));

    cudaMemcpyAsync(workerOffestAndLimit_dev, workerOffestAndLimit_host, sizeof(int)*16*noVars, cudaMemcpyHostToDevice,streams[3]);



    cudaMemcpyAsync(offset_dev, stream_buffer, sizeof(int)*noStreams, cudaMemcpyHostToDevice,streams[1]);

    divideInStrems(noBlocks,noBlocks_host);
    divideInStrems(currTableSize,CTsizes_host);
    divideInStrems((_supportSize/32)+1,ss32_host);

    lastStream_BL=0;
    lastStream_CT=0;
    lastStream_SS=0;
    for(int i=0;i<noStreams;i++){
        if(ss32_host[i]>0){
            lastStream_SS=i;
        }
        if(CTsizes_host[i]>0){
            lastStream_CT=i;
        }
        if(noBlocks_host[i]>0){
            lastStream_BL=i;
        }
    }
    cudaDeviceSynchronize();
    cudaFree(offset_host);
    cudaFree(stream_buffer);

}
void TableGPU::post(){
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void TableGPU::propagate(){
    enfoceGAC();
}

void TableGPU::enfoceGAC(){
    
    cudaMemcpyAsync(_currTable_dev, _currTable_host, currTableSize*sizeof(unsigned int), cudaMemcpyHostToDevice,streams[0]);
    
    _s_val.clear();
    _s_sup.clear();
    _s_sup.shrink_to_fit();
    _s_val.shrink_to_fit();
    int internalIndex=0;
    int output=0;

    for (int i = 0; i < _vars.size(); i++){
        //update s_val and the deltas
        if(_vars[i]->changed()){
            _s_val.push_back(i);
            _svSize_sval_host[internalIndex+1]=i;
            internalIndex++;
        }
        //update s_sup
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
        }
    }
    
    _svSize_sval_host[0]=_s_val.size();

    cudaMemcpyAsync(_svSize_sval_dev, _svSize_sval_host, sizeof(int)*(_s_val.size()+1), cudaMemcpyHostToDevice,streams[2]);

    dumpDomainsGPU();

    int offset=0;
    for(int i=0;i<=lastStream_SS;i++){
        cudaMemcpyAsync(_vars_dev+offset, _vars_host+offset, sizeof(unsigned int)*ss32_host[i], cudaMemcpyHostToDevice,streams[i]);
        offset=offset+ss32_host[i];
    }

  
    for(int i=0;i<currTableSize;i++){
        _currTable_host[i]=_currTable._words[i].value();
    }
    
    offset=0;
    for(int i=0;i<=lastStream_CT;i++){
        cudaMemcpyAsync(&_currTable_dev[offset], &_currTable_host[offset], CTsizes_host[i]*sizeof(unsigned int), cudaMemcpyHostToDevice,streams[i]);   
        offset=offset+CTsizes_host[i];
    }

  
    for(int i=0; i<=lastStream_BL; i++){
        updateTableGPU<<<noBlocks_host[i],256,256*sizeof(unsigned int),streams[i]>>>(_supports_dev,_svSize_sval_dev,_supportOffsetJmp_dev,_currTable_dev,_currTable_size_dev,_vars_dev,_output_dev,offset_dev+i,workerOffestAndLimit_dev);          
    }
    
    cudaDeviceSynchronize();
    //retrieve the output from the device
    cudaMemcpyAsync(_outputArray, _output_dev, sizeof(int)*noBlocks, cudaMemcpyDeviceToHost,streams[0]);


    
    
    cudaStreamSynchronize(streams[0]); 
    

    //performed on host, the number of blocks usually is small (e.g. if we have 1280 rows in the table we have 2 blocks)

    for(int i=0; i<noBlocks; i++){
        if(_outputArray[i]==1){
            output=1;
            break;
        }
    }

    if(output==1){
        failNow();
    }else{
        //we retrieve current table
        //getting back the current table
       
        cudaMemcpyAsync(_currTable_host, _currTable_dev, currTableSize*sizeof(unsigned int), cudaMemcpyDeviceToHost,streams[1]);
        cudaStreamSynchronize(streams[1]);

        //we need to update the current table

        _currTable.clearMask();
        _currTable.addToMaskArray(_currTable_host);
        
        _currTable.intersectWithMask();
        _currTable.clearMask();
      
        if(_currTable.isEmpty()){
            failNow();
        }
    }

    filterDomains();
}

void TableGPU::dumpDomainsGPU(){
    for(int i=0; i < _s_val.size(); ++i){

        int index=_s_val[i];

        int starting_word=(_supportOffsetJmp[index])/32;
        int words_to_reset=-1;
        int to=-1;
        if(index<noVars-1){
            to=_supportOffsetJmp[index+1]/32;
            words_to_reset=to-starting_word;
        }else{
            to=(_supportSize/32)+1;
            words_to_reset=to-starting_word;
            
        }
        for(int j=1;j<words_to_reset;j++){
            _vars_host[starting_word+j]=0;
        }
        
        if(words_to_reset>1){
            _vars_host[starting_word]=_vars_host[starting_word] & bitsFromLeft((_supportOffsetJmp[index])%32);
            
            if(index<noVars-1){
                _vars_host[starting_word+words_to_reset]=_vars_host[starting_word+words_to_reset] & bitsFromRight((32-_supportOffsetJmp[index+1])%32);
            }else{
                _vars_host[starting_word+words_to_reset]=0;
            }
        }else{
            //both masks on one word
            if(index<noVars-1){

                _vars_host[starting_word]=_vars_host[starting_word] & ( bitsFromLeft(_supportOffsetJmp[index]%32) | bitsFromRight((32-_supportOffsetJmp[index+1])%32));

            }else{
                _vars_host[starting_word]=_vars_host[starting_word] & bitsFromLeft((_supportOffsetJmp[index]%32));
            }

        }
        
        for (int j = _vars[index]->min(); j <= _vars[index]->max();  j++){ 

            if(_vars[index]->contains(j)){
                int wordIndex=(j-_variablesOffsets[index]+_supportOffsetJmp[index])/32;
                _vars_host[wordIndex]=_vars_host[wordIndex]|(0x80000000>>((_supportOffsetJmp[index]+j-_variablesOffsets[index])%32));

            }

        }
        
    }
}

int TableGPU::bitsFromRight(int n) {
    return (1 << (n)) - 1;
   
}
int TableGPU::bitsFromLeft(int n) {
    if (n == 0) return 0;       
    return ~0 << (32 - n);
}

// 1 th per support row
__global__ void updateTableGPU(unsigned int* _supports_dev,int * _svSize_off_sval_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* _currTable_dev_size, unsigned int* _vars_dev, int* output, int* offsetPerTh,int* offsetsAndLimits){


    int blockIdxx=blockIdx.x+(*offsetPerTh);

    
    int thPos = blockIdxx * 32 + threadIdx.x; //which currTable word we are considering

    int varIndex=0;
    extern __shared__ unsigned int mask[]; //mask (128)
    
    int th_tableFourth=threadIdx.x>>5; //0..3   mapped from th 0..127
    int th_mappedPos=threadIdx.x%32+(32*blockIdxx); //is thPos as if in the case we had 32 threads per block

    //clear mask MANDATORY

    mask[threadIdx.x]=0;
    
    if(th_mappedPos>=*_currTable_dev_size){
        return;
    }
    //the first th of each fourth, calculates how many iterations it will do
   
    //printf("%%%%%% GPU th %d actually looking at: %d, at fourth: %d \n",( blockIdx.x * blockDim.x + threadIdx.x),th_mappedPos,th_tableFourth);

    //each 32 threads will take care of a different variable
    for(int i=0; i<_svSize_off_sval_dev[0]; i++){
        
        varIndex=_svSize_off_sval_dev[i+1];
        int loops=((_supportOffsetJmp_dev[varIndex+1]-(_supportOffsetJmp_dev[varIndex])));    
    
        
        
        int from=_supportOffsetJmp_dev[varIndex];
        //printf("%%%%%% GPU var %d changed, loops for me: %d, in my case (thread %d) we do %d loops jumping from %d (accessing %d)\n",varIndex,mask[128+th_tableFourth],blockIdx.x * blockDim.x + threadIdx.x, mask[128+th_tableFourth],mask[128+th_tableFourth+4],128+th_tableFourth+4);
        //1/4 of the domain
        for(int j=0; j<offsetsAndLimits[varIndex*16+th_tableFourth]; j++){

            int wordIndex=(from+j+offsetsAndLimits[varIndex*16+th_tableFourth+8])/32; //row of supports
            int maskContains=1<<(31-j-_supportOffsetJmp_dev[varIndex]-offsetsAndLimits[varIndex*16+th_tableFourth+8]+wordIndex*32);

            if(_vars_dev[wordIndex] & maskContains){ //check if val in domain
                //printf("%%%%%% GPU INSIDE th %d var %d contains %d\n",thPos,varIndex,j);
                int off=(j+offsetsAndLimits[varIndex*16+th_tableFourth+8])*(*_currTable_dev_size)+(_supportOffsetJmp_dev[varIndex]*(*_currTable_dev_size))+threadIdx.x%32; //1 -> the size of the currTable
                mask[threadIdx.x]=mask[threadIdx.x] | _supports_dev[off];
            }            
        }
        __syncthreads();


        //printing complete mask
        //printf("%%%%%% GPU th %d complete mask for var %d is %u, table before[%d] %u\n",thPos,varIndex,mask[threadIdx.x],thPos,_currTable_dev[thPos]);
        if(th_tableFourth%2==0){
            //32 ths
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+32];
        }
        __syncthreads();
        if(th_tableFourth%4==0){
            //32 ths
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+64];
        }
        __syncthreads();
        if(th_tableFourth==0){
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+128];
            _currTable_dev[th_mappedPos]=mask[threadIdx.x] & _currTable_dev[th_mappedPos];   
        }
        mask[threadIdx.x]=0;
        //printf("%%%%%% ******* new var ******** \n");
    }
    
    if(threadIdx.x==0){
        for(int i=blockIdxx*32;i<(blockIdxx+1)*32;i++){
            if(_currTable_dev[i]!=0){

                //printf("%%%%%% GPU th %d kernel over \n",thPos);
                output[blockIdxx]=0;
                return;
            }
        }
        output[blockIdxx]=1;
    }

}




//utilities to remove



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
void TableGPU::printBits(unsigned int num) {
    // Extracting each bit of the int and printing it
    //yes rather weird function, but since we need to print %%%%%
    char str[32] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
    for (int i = 31; i >= 0; i--) {
        str[i] = (num >> i) & 1; 
        printf("%d",str[i]);
    }

    printf(" \n");
    
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


void TableGPU::divideInStrems(int size,int * where) {
    
    for (int i = 0; i < noStreams; ++i) {
        where[i] = size / noStreams;              // Divide the size by the no of streams
    }
    int remainder = size % noStreams;           // Calculate the remainder

    // Distribute the remainder across the first few parts
    for (int i = 0; i < remainder; ++i) {
        where[i]++;
    }
}

void TableGPU::varOffsetLimit(int size,int * where) {
    
    for (int i = 0; i < 8; ++i) {
        where[i] = size / 8;              // Divide the size by the no of streams
    }
    int remainder = size % 8;           // Calculate the remainder

    // Distribute the remainder across the first few parts
    for (int i = 0; i < remainder; ++i) {
        where[i]++;
    }


    where[8]=0;
    where[9]=where[0];
    where[10]=where[1]+where[9];
    where[11]=where[2]+where[10];
    where[12]=where[3]+where[11];
    where[13]=where[4]+where[12];
    where[14]=where[5]+where[13];
    where[15]=where[6]+where[14];

}
