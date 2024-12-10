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

    streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*noStreams);
    printf("%%%%%% creating %d streams\n",noStreams);



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
        updateTableGPU<<<noBlocks_host[i],128,(128+8)*sizeof(unsigned int),streams[i]>>>(_supports_dev,_svSize_sval_dev,_supportOffsetJmp_dev,_currTable_dev,_currTable_size_dev,_vars_dev,_output_dev,offset_dev+i);          
    }
    
    cudaDeviceSynchronize();
    fflush(stdout);
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
__global__ void updateTableGPU(unsigned int* _supports_dev,int * _svSize_off_sval_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* _currTable_dev_size, unsigned int* _vars_dev, int* output, int* offsetPerTh){


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
        
        __syncthreads();
        if(th_mappedPos==0){
            mask[128+th_tableFourth] = loops / 4;
            int remainder = loops % 4;          
            
            mask[128+th_tableFourth]+=(th_tableFourth<remainder);
            
            mask[128+th_tableFourth+4]=0;
            
            if(threadIdx.x==0){
                //printf("%%%%%% thread setting masks\n");
                mask[133]=mask[128];
                mask[134]=mask[133]+mask[129];
                mask[135]=mask[134]+mask[130];
                //printf("%%%%%% mask[132]: %d,mask[133]: %d, mask[134]: %d, mask[135]: %d\n",mask[132],mask[133],mask[134],mask[135]);
            }            
        }
        //printf("%%%%%% th %d comunque tutti prima di loops for me \n",blockIdx.x * blockDim.x + threadIdx.x);
        __syncthreads();
        
        int from=_supportOffsetJmp_dev[varIndex];
        //printf("%%%%%% GPU var %d changed, loops for me: %d, in my case (thread %d) we do %d loops jumping from %d (accessing %d)\n",varIndex,mask[128+th_tableFourth],blockIdx.x * blockDim.x + threadIdx.x, mask[128+th_tableFourth],mask[128+th_tableFourth+4],128+th_tableFourth+4);
        //1/4 of the domain
        for(int j=0; j<mask[128+th_tableFourth]; j++){

            int wordIndex=(from+j+mask[128+th_tableFourth+4])/32; //row of supports
            int maskContains=1<<(31-j-_supportOffsetJmp_dev[varIndex]-mask[128+th_tableFourth+4]+wordIndex*32);

            //printf("%%%%%% GPU th %d var %d, accessing ctWord %d, checking mask %d\n",(thPos),varIndex,wordIndex,maskContains);
            //printf("%%%%%% GPU th %d var %d, accessing word %d, maskContains: %u\n",thPos,varIndex,wordIndex, maskContains);
            //printf("%%%%%% GPU th %d var %d maskContains %u\n",thPos,varIndex,maskContains);
            if(_vars_dev[wordIndex] & maskContains){ //check if val in domain
                //printf("%%%%%% GPU INSIDE th %d var %d contains %d\n",thPos,varIndex,j);
                int off=(j+mask[128+th_tableFourth+4])*(*_currTable_dev_size)+(_supportOffsetJmp_dev[varIndex]*(*_currTable_dev_size))+threadIdx.x%32; //1 -> the size of the currTable
                //printf("%%%%%% GPU INSIDE th %d off %d, _currTable_dev_size: %u,_supportOffsetJmp_dev[varIndex]: %d\n",thPos,off,*_currTable_dev_size,_supportOffsetJmp_dev[varIndex]);
                //printf("%%%%%% GPU th %d, var %d contains %d accessing the %d support word\n",thPos,varIndex,maskContains,off);
                mask[threadIdx.x]=mask[threadIdx.x] | _supports_dev[off];
                //printf("%%%%%% GPU INSIDE th %d mask related to var %d is %u, only the mask %u (accessing %d)\n",thPos,varIndex,mask[threadIdx.x],_supports_dev[off],off);
            }
        }

        __syncthreads();
        //printing complete mask
        //printf("%%%%%% GPU th %d complete mask for var %d is %u, table before[%d] %u\n",thPos,varIndex,mask[threadIdx.x],thPos,_currTable_dev[thPos]);
        if(th_tableFourth==0){
            //32 ths
            for(int i=0;i<4;i++){
                mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+i*32];
                //printf("%%%%%% GPU th %d merging with block %d, mask %u, currTable[%d] %u\n",thPos,i,mask[threadIdx.x],thPos,_currTable_dev[thPos]);
            }
            _currTable_dev[th_mappedPos]=mask[threadIdx.x] & _currTable_dev[th_mappedPos];   
            //printf("%%%%%% GPU th %d, mapped to %d ct is %u",thPos,th_mappedPos,_currTable_dev[th_mappedPos]);
        }

        __syncthreads();
        mask[threadIdx.x]=0;
        //printf("%%%%%% ******* new var ******** \n");
    }
    
    //printf("%%%%%% GPU th %d currTable[%d] %d\n",thPos,thPos,_currTable_dev[thPos]);
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
    //printf("%%%%%% GPU th %d kernel over \n",thPos);

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
//thanks