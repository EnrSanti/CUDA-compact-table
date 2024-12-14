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
    _stream_offset_dev=mallocDevice<int>(sizeof(int)*noStreams);
    workerOffestAndLimit_dev=mallocDevice<int>(sizeof(int)*32*noVars);
    
    
    printf("%%%%%% TableGPU constructor CT SIZE: %d \n",currTableSize);

    //on host side we create simpler structures to then copy the data
    int * offset_host;
    cudaMallocHost((void**)&_currTable_host, sizeof(unsigned int)*currTableSize);
    cudaMallocHost((void**)&_vars_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&_svSize_sval_host,sizeof(int)*(noVars+1));
    cudaMallocHost((void**)&offset_host,sizeof(int)*noStreams);
    cudaMallocHost((void**)&stream_buffer,sizeof(int)*noStreams);

    cudaMallocHost((void**)&CTsizes_host,sizeof(int)*noStreams);
    cudaMallocHost((void**)&ss32_host,sizeof(int)*noStreams);
    cudaMallocHost((void**)&noBlocks_host,sizeof(int)*noStreams);
    cudaMallocHost((void**)&workerOffestAndLimit_host,sizeof(int)*32*noVars);


    streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*noStreams);


    printf("%%%%%% TableGPU constructor support size: %d\n",_supportSize);
    for(int i=0;i<((_supportSize/32)+1);i++){
        _vars_host[i]=0xffffffff;
    }
    if(_supportSize%32!=0)
        _vars_host[(_supportSize/32)]=0xffffffff<<(32-(_supportSize%32));

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
    cudaMemcpyAsync(_supportSize_dev, &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_variablesOffsets_dev, _variablesOffsets.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supportOffsetJmp_dev, _supportOffsetJmp.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(&_supportOffsetJmp_dev[noVars], &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_currTable_size_dev, &currTableSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);


    //compute once and transfer the offsets for the streams:
    noBlocks=(currTableSize/8)+1;
    divideInStrems(noBlocks,offset_host);
    stream_buffer[0]=0;
    for(int i=1; i<noStreams; i++){
        if(offset_host[i]>0){
            stream_buffer[i]=stream_buffer[i-1]+offset_host[i-1];
        }
    }
    

    for(int i=0;i<noVars-1;i++){
        varOffsetLimit(_supportOffsetJmp[i+1]-_supportOffsetJmp[i],workerOffestAndLimit_host+(i*32));
    }
    varOffsetLimit(_supportSize-_supportOffsetJmp[noVars-1],workerOffestAndLimit_host+((noVars-1)*32));

    cudaMemcpyAsync(workerOffestAndLimit_dev, workerOffestAndLimit_host, sizeof(int)*32*noVars, cudaMemcpyHostToDevice,streams[0]);


    printf("%%%%%% TableGPU streamoffsetDev[0]: %d  streamoffsetDev[1]: %d \n",stream_buffer[0],stream_buffer[1]);
    cudaMemcpyAsync(_stream_offset_dev, stream_buffer, sizeof(int)*noStreams, cudaMemcpyHostToDevice,streams[0]);

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
    printf("%%%%%% dumping domains\n");
    for(int i=0;i<(_supportSize/32)+1;i++){
        printf("%%%%%% [%d] ",i);
        printBits(_vars_host[i]);
    }

}
void TableGPU::post(){
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void TableGPU::propagate(){
    enfoceGAC();
}
void TableGPU::enfGACDev(){
    cudaMemcpyAsync(_currTable_dev, _currTable_host, currTableSize*sizeof(unsigned int), cudaMemcpyHostToDevice,streams[0]);

    int output=0;
    
    _svSize_sval_host[0]=_s_val.size();

    cudaMemcpyAsync(_svSize_sval_dev, _svSize_sval_host, sizeof(int)*(_s_val.size()+1), cudaMemcpyHostToDevice,streams[0]);

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
        //pass: the supports, the changed variables + how many, the indexes for the support, the table and the size, the domains, the stream offset and 32*vars ints which tells what range of the varialbe to check according to the index of the th
        printf("%%%%%% stream[i] %d, launching %d blocks\n",i,noBlocks_host[i]);
        updateTableGPU<<<noBlocks_host[i],128,128*sizeof(unsigned int),streams[i]>>>(_supports_dev,_svSize_sval_dev,_supportOffsetJmp_dev,_currTable_dev,_currTable_size_dev,_vars_dev,_stream_offset_dev+i,workerOffestAndLimit_dev);          
    }
    
    cudaDeviceSynchronize();

    cudaMemcpyAsync(_currTable_host, _currTable_dev, currTableSize*sizeof(unsigned int), cudaMemcpyDeviceToHost,streams[0]);

    _currTable.clearMask();
    
    cudaDeviceSynchronize();

    //we need to update the current table

    _currTable.addToMaskArray(_currTable_host);
    
    _currTable.intersectWithMask();
    _currTable.clearMask();

    if(_currTable.isEmpty()){
        failNow();
    }

}
void TableGPU::enfoceGAC(){
    
    _s_val.clear();
    _s_sup.clear();
    _s_sup.shrink_to_fit();
    _s_val.shrink_to_fit();
    int internalIndex=0;
    
   int overallSize=0;
    for (int i = 0; i < _vars.size(); i++){
        //update s_val and the deltas
        if(_vars[i]->changed()){
            _s_val.push_back(i);
            _svSize_sval_host[internalIndex+1]=i;
            internalIndex++;
            overallSize=overallSize+_vars[i]->intialSize();
            printf("%%%%%% var %d changed\n",i);
            //print all the values
            for (int j = _vars[i]->initialMin(); j <= _vars[i]->initialMax();  j++){
                if(_vars[i]->contains(j))
                    printf("%%%%%% var %d value %d in domain\n",i,j);
                else
                    printf("%%%%%% var %d value %d not in domain\n",i,j);
                
            }
        }
        //update s_sup
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
        }
    }
    fflush(stdout);


    //overallSize>4000
    //if(overallSize>2000){ //to better see advantages when testing remove and do only enfGACDev();
        enfGACDev();
    //}else{
        //updateTable();
    //}

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
        printf("%%%%%%  var changed: %d, it starts at %d and ends at %d \n",index,starting_word,to);
       
        for(int j=1;j<words_to_reset;j++){
            _vars_host[starting_word+j]=0;
        }
        
        if(words_to_reset>=1){
            _vars_host[starting_word]=_vars_host[starting_word] & bitsFromLeft((_supportOffsetJmp[index])%32);
            
            if(index<noVars-1){
                _vars_host[starting_word+words_to_reset]=_vars_host[starting_word+words_to_reset] & bitsFromRight((32-_supportOffsetJmp[index+1])%32);
            }else{
                _vars_host[starting_word+words_to_reset]=0;
            }
        }else{
            //both masks on one word
            
            if(index<noVars-1){
                printf("%%%%%%  var changed: 0 words to reset, I AM IN THEN, index: %d before: %d maskLeft: %d maskRight: %d \n",starting_word, _vars_host[starting_word], bitsFromLeft(_supportOffsetJmp[index]%32),bitsFromRight((32-_supportOffsetJmp[index+1])%32));
                _vars_host[starting_word]=_vars_host[starting_word] & ( bitsFromLeft(_supportOffsetJmp[index]%32) | bitsFromRight((32-_supportOffsetJmp[index+1])%32));

            }else{
                printf("%%%%%%  var changed: 0 words to reset, I AM IN ELSE");
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
    printf("%%%%%% dumping domains\n");
    for(int i=0;i<(_supportSize/32)+1;i++){
        printf("%%%%%% [%d] ",i);
        printBits(_vars_host[i]);
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
__global__ void updateTableGPU(unsigned int* _supports_dev,int * _svSize_off_sval_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* _currTable_dev_size, unsigned int* _vars_dev, int* offsetPerTh,int* offsetsAndLimits){


    extern __shared__ unsigned int mask[]; //mask (128 ints)

    //block: actual block in the stream + how many are before me in other streams
    int blockIdxx=blockIdx.x+(*offsetPerTh);
    int rowsPerBlock=16;
    int colsPerBlock=8;
    int varIndex=0;
    
    int th_col=threadIdx.x%8; //from 0..127 to 0..7 (which column do we look at), each block looks at 8 columns, groups of 16 threads (size 8) will share the same column
    int th_row=threadIdx.x/8; //from 0..127 to 0..15 (which row do i am part of), 8 threads will share the same row

    int th_mappedPos_stream=threadIdx.x%colsPerBlock+(colsPerBlock*blockIdxx); //it is thPos as if I didn't have to consider the other streams
    //groups of 16 threads will share the same position

    //clear mask MANDATORY
    mask[threadIdx.x]=0;
    

    if(th_mappedPos_stream>=*_currTable_dev_size){
        return;
    }

    //each 32 threads will take care of the same var
    for(int i=0; i<_svSize_off_sval_dev[0]; i++){
        
        //variable index
        varIndex=_svSize_off_sval_dev[i+1];

        //the starting point of the supports for the var
        int from=_supportOffsetJmp_dev[varIndex];
        int iterations16_th=varIndex*32+th_row; //16 values, equal for groups of 8 threads
        int offset16_th=offsetsAndLimits[iterations16_th+16]; //16 values, equal for groups of 8 threads

        //1/16 of the domain, from 0 to the upper bound of each group of 16 threads
        for(int j=0; j<offsetsAndLimits[iterations16_th]; j++){

            int wordIndex=(from+j+offset16_th)/32; //piece of row of supports, not the cell, the row piece of row the block looks at
            int maskContains=1<<(31-j-_supportOffsetJmp_dev[varIndex]-offset16_th+wordIndex*32);

            if((_vars_dev[wordIndex] & maskContains) != 0){ //check if val in domain
                //off is != for each of the 128 ths
                int off=(j+offset16_th)*(*_currTable_dev_size)+(_supportOffsetJmp_dev[varIndex]*(*_currTable_dev_size))+threadIdx.x%8; 

                mask[threadIdx.x]=mask[threadIdx.x] | _supports_dev[off];
                printf("%%%%%% block %d MODIFYING (c: %d, r: %d th %d, mapped pos %d, looking at bit %d) which is in the domain, offset %d (word del supporto) e thidx %d, mask at the end %d, vardev[%d]: %d maskContains %d \n",blockIdx.x,th_col,th_row,threadIdx.x, th_mappedPos_stream,(from+j+offset16_th), off, threadIdx.x,mask[threadIdx.x],wordIndex,_vars_dev[wordIndex],maskContains);

            }            
        }
        __syncthreads();


        printf("%%%%%% block %d AT THE END all ths, GPU th c %d r %d MASK[%d] %d \n",blockIdx.x,th_col,th_row,threadIdx.x,mask[threadIdx.x]);
        //printing complete mask
        //printf("%%%%%% GPU th %d complete mask for var %d is %u, table before[%d] %u\n",thPos,varIndex,mask[threadIdx.x],thPos,_currTable_dev[thPos]);
        if(threadIdx.x<64){
            //64 ths
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+64];
            printf("%%%%%% block %d AT THE END partial 64 GPU th c %d r %d, MASK[%d] %d \n",blockIdx.x,th_col,th_row,threadIdx.x,mask[threadIdx.x]);
        }
        __syncthreads();
        if(threadIdx.x<32){
            //32 ths
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+32];
            printf("%%%%%% block %d AT THE END partial 32 GPU th c %d r %d MASK %d \n",blockIdx.x,th_col,th_row,mask[threadIdx.x]);
        }
        __syncthreads();
        if(threadIdx.x<16){
            //32 ths
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+16];

            printf("%%%%%% block %d AT THE END partial 16 GPU th c %d r %d MASK %d \n",blockIdx.x,th_col,th_row,mask[threadIdx.x]);
        }
        __syncthreads();
        //8 threads to this last operation
        if(threadIdx.x<8){
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+8];

            printf("%%%%%% block %d FINE 1, AT THE END, GPU th row %d col %d ct BEFORE %d, MASK %d \n",blockIdx.x,th_row,th_col,_currTable_dev[th_mappedPos_stream],mask[threadIdx.x]);
            _currTable_dev[th_mappedPos_stream]=mask[threadIdx.x] & _currTable_dev[th_mappedPos_stream];   
            printf("%%%%%% block %d FINE 2, GPU th  col %d row %d (mapped %d, non ci dovrebbero essere altri numeri uguali) ct %d \n",blockIdx.x,th_row,th_col,th_mappedPos_stream,_currTable_dev[th_mappedPos_stream]);
        }
        mask[threadIdx.x]=0;
    }
    


}




__global__ void printGPUdata(int *_supportSize_dev, int *_variablesOffsets_dev,unsigned int *_currTable_dev,unsigned int *_supports_dev,int * _supportOffsetJmp_dev, int* currTable_size_dev, int* offsetsAndLimits){
    printf("%%%%%% -------------------------- printGPUdata -------------------------- \n");
    printf("%%%%%% threadIdx.x: %d\n",threadIdx.x);
    printf("%%%%%% _supportSize_dev: %d\n",*_supportSize_dev);
    //printing the offsets
    int k=0;
    int off=0;
    
    printf("%%%%%% currTable\n");
    for(int j=0;j<*currTable_size_dev;j++){
        printf("%%%%%% [%d] ", j);
        printBitsGPU(_currTable_dev[j]);
    }
    printf("%%%%%% limits and offsets\n");
    for(int j=0;j<3;j++){
    
        for(int i=0;i<32;i++){
            printf("%%%%%% [%d] %d \n", i,offsetsAndLimits[i]);
        }
        printf("%%%%%% --------------------\n");

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
    
    for (int i = 0; i < 16; ++i) {
        where[i] = size / 16;              // Divide the size by the no of streams
    }
    int remainder = size % 16;           // Calculate the remainder

    // Distribute the remainder across the first few parts
    for (int i = 0; i < remainder; ++i) {
        where[i]++;
    }

    /*
    where[8]=0;
    where[9]=where[0];
    where[10]=where[1]+where[9];
    where[11]=where[2]+where[10];
    where[12]=where[3]+where[11];
    where[13]=where[4]+where[12];
    where[14]=where[5]+where[13];
    where[15]=where[6]+where[14];
    where[16]=where[7]+where[15];
    */
    //upper bound


    where[16]=0;
    where[17]=where[0];
    where[18]=where[1]+where[17];
    where[19]=where[2]+where[18];
    where[20]=where[3]+where[19];
    where[21]=where[4]+where[20];
    where[22]=where[5]+where[21];
    where[23]=where[6]+where[22];
    where[24]=where[7]+where[23];
    where[25]=where[8]+where[24];
    where[26]=where[9]+where[25];
    where[27]=where[10]+where[26];
    where[28]=where[11]+where[27];
    where[29]=where[12]+where[28];
    where[30]=where[13]+where[29];
    where[31]=where[14]+where[30];


}
__global__ void isEmpty(unsigned int* _currTable_dev,int* _currTable_size_dev, int* res){
   extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (idx < *_currTable_size_dev) ? (_currTable_dev[idx] != 0) : 0; // Store 1 if the element is non-zero
    __syncthreads();

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] |= sdata[tid + stride]; // OR operation to detect any non-zero value
        }
        __syncthreads();
    }

    // Write result of this block to global memory
    if (tid == 0) {
        atomicOr(res, sdata[0]);
    }
}