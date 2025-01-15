#include "gpu_constriants/table.cuh"

#include <chrono>


TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){

    //auto start = std::chrono::high_resolution_clock::now();
    int noTuples=tuples.size();
    noVars=vars.size();
    
    currTableSize=(noTuples/32)+1; 
    
    

    // Memory allocation

    //auto start2 = std::chrono::high_resolution_clock::now();
    
    cudaMalloc((void**)&_noVars_dev, sizeof(int));
    cudaMalloc((void**)&_CT_MASKCT_svSize_sval_sSize_sSup_dev, sizeof(unsigned int)*(2*currTableSize+2*noVars+2));
    cudaMalloc((void**)&_supports_dev, sizeof(unsigned int)*_supportSize*currTableSize);
    cudaMalloc((void**)&_supportSize_dev, sizeof(int));
    cudaMalloc((void**)&_variablesOffsets_dev, sizeof(int)*noVars);
    cudaMalloc((void**)&_supportOffsetJmp_dev, sizeof(int)*(noVars+1));
    cudaMalloc((void**)&_currTable_size_dev, sizeof(int));
    cudaMalloc((void**)&_vars_dev, sizeof(int)*((_supportSize/32)+1)); //matrix
    
    cudaMalloc((void**)&workerOffestAndLimit_dev, sizeof(int)*64*noVars);
    
    
    //auto end2 = std::chrono::high_resolution_clock::now();
    //auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    //printf("%%%%%% Time taken for CUDAMALLOC: %ld microseconds\n", duration2.count());
    
    

    //start2 = std::chrono::high_resolution_clock::now();
    
    //on host side we create simpler structures to then copy the data

    cudaMallocHost((void**)&_CT_MASKCT_svSize_sval_sSize_sSup_host, sizeof(unsigned int)*2*(noVars+1+currTableSize));
    cudaMallocHost((void**)&_vars_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&_vars_to_remove_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&workerOffestAndLimit_host,sizeof(int)*64*noVars);


    //dumped with calloc
    dumped=(bool*)calloc(noVars,sizeof(bool));


    //end2 = std::chrono::high_resolution_clock::now();
    //duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    //printf("%%%%%% Time taken for CUDAMALLOC (HOST): %ld microseconds\n", duration2.count());
    

    streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*noStreams);

    cudaError_t err = cudaStreamCreate(&streams[0]);
    
    

    for(int i=0;i<noVars-1;i++){
        varOffsetLimit(_supportOffsetJmp[i+1]-_supportOffsetJmp[i],workerOffestAndLimit_host+(i*64));
    }
    varOffsetLimit(_supportSize-_supportOffsetJmp[noVars-1],workerOffestAndLimit_host+((noVars-1)*64));



    cudaMemcpyAsync(workerOffestAndLimit_dev, workerOffestAndLimit_host, sizeof(int)*64*noVars, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_noVars_dev, &noVars, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supports_dev, _supports, sizeof(unsigned int)*_supportSize*currTableSize, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supportSize_dev, &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_variablesOffsets_dev, _variablesOffsets.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supportOffsetJmp_dev, _supportOffsetJmp.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(&_supportOffsetJmp_dev[noVars], &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_currTable_size_dev, &currTableSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);


    int buffSize=0;   
    for(int i=0;i<noVars;i++){
        buffSize=max(buffSize,vars[i]->size()/32+2);
    }
    buffer=(unsigned int*)calloc(buffSize,sizeof(unsigned int));

    //end2 = std::chrono::high_resolution_clock::now();
    //duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    //printf("%%%%%% Time taken for CUDA cpy: %ld microseconds\n", duration2.count());

    //compute once and transfer the offsets for the streams:
    noBlocks=(currTableSize/4)+1;
    noBlocksFilter=((_supportSize/32)+1);

    setAsynchronous(true);
    cudaStreamSynchronize(streams[0]);

    
    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //printf("%%%%%% Time taken INIT CUDA: %ld microseconds\n", duration.count());
    

}
void TableGPU::post(){
    propagate();
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void TableGPU::propagate(){

    //printf("%%%%%% ::::::::::::::::::::::::::::::::::::: \n");
    //auto start = std::chrono::high_resolution_clock::now();
    offload();
    retrieve();

    //auto end = std::chrono::high_resolution_clock::now();
    
    
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //printf("%%%%%% Time taken enfGAC: %ld microseconds\n", duration.count());
    //fflush(stdout);
    
}
void TableGPU::offload(){
    _s_val.clear();
    _s_val.shrink_to_fit();

    _s_sup.clear();
    _s_sup.shrink_to_fit();

    int internalIndex=currTableSize*2+2;
    
    //int overallSize=0;
    for (int i = 0; i < _vars.size(); i++){
        //update s_val and the deltas
        if(_vars[i]->changed()){
            _s_val.push_back(i);
            _CT_MASKCT_svSize_sval_sSize_sSup_host[internalIndex]=i;
            internalIndex++;
        }
    }

    for (int i = 0; i < _vars.size(); i++){
        //update s_sup
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
            _CT_MASKCT_svSize_sval_sSize_sSup_host[internalIndex]=i;
            internalIndex++;
        }
    }

    //for each var in the table add it to ssup vector
  
    _CT_MASKCT_svSize_sval_sSize_sSup_host[currTableSize*2]=_s_val.size();
    _CT_MASKCT_svSize_sval_sSize_sSup_host[currTableSize*2+1]=_s_sup.size();

    for(int i=0;i<currTableSize;i++){
        _CT_MASKCT_svSize_sval_sSize_sSup_host[i]=_currTable._words[i].value();
    }
    
    //aggiungi pure ct ua
    cudaMemcpyAsync(_CT_MASKCT_svSize_sval_sSize_sSup_dev, _CT_MASKCT_svSize_sval_sSize_sSup_host, sizeof(unsigned int)*(2*currTableSize+_s_val.size()+_s_sup.size()+2), cudaMemcpyHostToDevice,streams[0]);   
    
    //metti dump domini qui
    dumpDomainsGPU2();

    cudaMemcpyAsync(_vars_dev, _vars_host, sizeof(int)*((_supportSize/32)+1), cudaMemcpyHostToDevice,streams[0]);


    //pass: the supports, the changed variables + how many, the indexes for the support, the table and the size, the domains,  and 32*vars ints which tells what range of the varialbe to check according to the index of the th (modifies CT with CT & mask)
    updateTableGPU<<<noBlocks,128,128*sizeof(unsigned int),streams[0]>>>(_supports_dev,_CT_MASKCT_svSize_sval_sSize_sSup_dev+(2*currTableSize),_supportOffsetJmp_dev,_CT_MASKCT_svSize_sval_sSize_sSup_dev,_currTable_size_dev,_vars_dev,workerOffestAndLimit_dev);          
    

    //si copio mask in ct per semplcità
    cudaMemcpyAsync(_CT_MASKCT_svSize_sval_sSize_sSup_host, _CT_MASKCT_svSize_sval_sSize_sSup_dev, currTableSize*sizeof(unsigned int), cudaMemcpyDeviceToHost,streams[0]);

    filterDomainsGPU<<<noBlocksFilter,32,32*sizeof(unsigned int),streams[0]>>>(_CT_MASKCT_svSize_sval_sSize_sSup_dev,_currTable_size_dev,_vars_dev,_supportOffsetJmp_dev,_supports_dev, _supportSize_dev);
    //launch filtering 

}
void TableGPU::retrieve(){

    cudaStreamSynchronize(streams[0]);
    cudaMemcpyAsync(_vars_to_remove_host, _vars_dev, sizeof(int)*((_supportSize/32)+1), cudaMemcpyDeviceToHost,streams[0]);

    _currTable.addToMaskArray(_CT_MASKCT_svSize_sval_sSize_sSup_host);
    
    _currTable.intersectWithMask();
    _currTable.clearMask();

    if(_currTable.isEmpty()){
        //sync stream 0
        failNow();
    }


    //copy back the domains
    
    cudaStreamSynchronize(streams[0]);
    
    //for all the vars in ssup
    for(int i=0;i<_s_sup.size();i++){

        int index=_s_sup[i];
        int starting_word=(_supportOffsetJmp[index]+(_vars[index]->min()-_vars[index]->initialMin()))/32;
        int starting_bit=(_supportOffsetJmp[index]+(_vars[index]->min()-_vars[index]->initialMin()))%32; 
        
        //from the min to the max (can be changed);
        for (int j = _vars[index]->min(); j <= _vars[index]->max();  j++){ 
            if((_vars_to_remove_host[starting_word] & (0x80000000>>starting_bit))!=0){
                _vars[index]->remove(j);
            }
            starting_bit++;
            if(starting_bit==32){
                starting_bit=0;
                starting_word++;
            }
        }
    }
}
void TableGPU::dumpDomainsGPU2(){

    //auto start = std::chrono::high_resolution_clock::now();
    for(int index=0; index < noVars; index++){
        //quali variaibli skip

        if(!(_vars[index]->changed()) && _vars[index]->size()==1)
            continue;

        if(dumped[index] && !(_vars[index]->changed()))
            continue;
        
        
        
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


        if(words_to_reset>=1){
            _vars_host[starting_word]=_vars_host[starting_word] & bitsFromLeft((_supportOffsetJmp[index])%32);
            
            if(index<noVars-1){
                _vars_host[starting_word+words_to_reset]=_vars_host[starting_word+words_to_reset] & bitsFromRight((32-_supportOffsetJmp[index+1] % 32 + 32)%32);
            }else{
                _vars_host[starting_word+words_to_reset]=0;
            }
        }else{
            //both masks on one word
            
            if(index<noVars-1){
                _vars_host[starting_word]=_vars_host[starting_word] & ( bitsFromLeft(_supportOffsetJmp[index]%32) | bitsFromRight((32-_supportOffsetJmp[index+1] % 32 + 32)%32));

            }else{
                _vars_host[starting_word]=_vars_host[starting_word] & bitsFromLeft((_supportOffsetJmp[index]%32));
            }

        }

        starting_word=(_supportOffsetJmp[index]-_variablesOffsets[index]+_vars[index]->min())/32;
        int ending_word=(_supportOffsetJmp[index]-_variablesOffsets[index]+_vars[index]->max())/32;
        words_to_reset=ending_word-starting_word;
        int starting_bit=(_supportOffsetJmp[index]-_variablesOffsets[index]+_vars[index]->min())%32;
        int ending_bit=(_variablesOffsets[index]+_vars[index]->max()-_variablesOffsets[index])%32;
        


        int maskRight=0;
        int maskLeft=0;
        if(starting_word==((_supportOffsetJmp[index]-_variablesOffsets[index]+_vars[index]->initialMin())/32))
            maskLeft=bitsFromLeft((_supportOffsetJmp[index]-_variablesOffsets[index]+_vars[index]->initialMin())%32);

        if(ending_word==((_supportOffsetJmp[index]-_variablesOffsets[index]+_vars[index]->initialMax())/32))
            maskRight=bitsFromRight(31-(_supportOffsetJmp[index]-_variablesOffsets[index]+_vars[index]->initialMax())%32);
        
        for(int i=0;i<words_to_reset;i++){
            buffer[i]=0;
        }


        _vars[index]->dumpWithOffset(_vars[index]->min(),_vars[index]->max(),buffer,starting_bit);
        
        
        if(words_to_reset>=1){
            _vars_host[starting_word]=buffer[0] | (_vars_host[starting_word] & maskLeft);
            int index2=1;
            for(int j=starting_word+1; j<ending_word; j++){
                _vars_host[j]=buffer[index2];
                index2++;
            }
            _vars_host[ending_word]=buffer[index2] | (_vars_host[ending_word] & maskRight);

        }else{   
            _vars_host[starting_word]=buffer[0] | (_vars_host[starting_word] & (maskLeft | maskRight));
        }
        dumped[index]=true;
    }

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //printf("%%%%%% Time of which by dump 2: %ld microseconds\n", duration.count());
    //fflush(stdout);
    
}
// 1 th per support row
__global__ void updateTableGPU(unsigned int* _supports_dev,unsigned int * _svSize_off_sval_dev, int *_supportOffsetJmp_dev, unsigned int * _CT_mask_dev,int* _currTable_dev_size, int* _vars_dev, int* offsetsAndLimits){


    extern __shared__ unsigned int mask[]; //mask (128 ints)

    //block: actual block in the stream + how many are before me in other streams
    int blockIdxx=blockIdx.x;
    int colsPerBlock=4;
    int varIndex=0;
    int th_col=threadIdx.x%4; //from 0..127 to 0..4 (which column do we look at), each block looks at 4 columns, groups of 32 threads (size 4) will share the same column
    int th_row=threadIdx.x/4; //from 0..127 to 0..31 (which row do i am part of), 32 threads will share the same row

    int th_mappedPos_stream=threadIdx.x%colsPerBlock+(colsPerBlock*blockIdxx); //it is thPos as if I didn't have to consider the other streams
  
    //groups of 32 threads will share the same position

    //clear mask MANDATORY
    mask[threadIdx.x]=0;
    

    if(th_mappedPos_stream>=*_currTable_dev_size){
        return;
    }
    //each 32 threads will take care of the same var
    for(int i=0; i<_svSize_off_sval_dev[0]; i++){
        
        //variable index
        varIndex=_svSize_off_sval_dev[i+2];

        //printf("%%%%%% sv size: _svSize_off_sval_dev[0]: %d %d \n",_svSize_off_sval_dev[0],varIndex);
        //the starting point of the supports for the var
        int from=_supportOffsetJmp_dev[varIndex];
        int iterations32_th=varIndex*64+th_row; //32 values, equal for 4 groups of threads
        int lastVal=varIndex*64+31;
        int offset32_th=offsetsAndLimits[iterations32_th+32]; //32 values, equal for groups of 4 threads

        //1/16 of the domain, from 0 to the upper bound of each group of 16 threads
        for(int j=0; j<offsetsAndLimits[lastVal]; j++){

            int wordIndex=(from+j+offset32_th)/32; //piece of row of supports, not the cell, the row piece of row the block looks at
            int maskContains=1<<(31-j-_supportOffsetJmp_dev[varIndex]-offset32_th+wordIndex*32);

            if((_vars_dev[wordIndex] & maskContains) != 0){ //check if val in domain
                //off is != for each of the 128 ths
                //deve tener conto del 
                int off=(j+offset32_th)*(*_currTable_dev_size)+(_supportOffsetJmp_dev[varIndex]*(*_currTable_dev_size))+blockIdxx*4+th_col; 

                mask[threadIdx.x]=mask[threadIdx.x] | _supports_dev[off];
            }  
            __syncthreads();          
        }
        //c'è un unroll sull'ultimo ciclo per poter inserire in syncThreads sopra
        if(offsetsAndLimits[lastVal]<offsetsAndLimits[iterations32_th]){
            int j=offsetsAndLimits[iterations32_th]-1;
            int wordIndex=(from+j+offset32_th)/32; //piece of row of supports, not the cell, the row piece of row the block looks at
            int maskContains=1<<(31-j-_supportOffsetJmp_dev[varIndex]-offset32_th+wordIndex*32);

            if((_vars_dev[wordIndex] & maskContains) != 0){ //check if val in domain
                //off is != for each of the 128 ths
                int off=(j+offset32_th)*(*_currTable_dev_size)+(_supportOffsetJmp_dev[varIndex]*(*_currTable_dev_size))+blockIdxx*4+th_col; 

                mask[threadIdx.x]=mask[threadIdx.x] | _supports_dev[off];
             }            
        }
        __syncthreads();


        if(threadIdx.x<64){
            //64 ths
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+64];
        }
        __syncthreads();
        if(threadIdx.x<32){
            //32 thscurrTableSize
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+32];
        }
        __syncthreads();
        if(threadIdx.x<16){
            //16 ths
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+16];

       }
        __syncthreads();
        if(threadIdx.x<8){
            //8 ths
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+8];

        }
        __syncthreads();
        //4 threads to this last operation
        if(threadIdx.x<4){
            mask[threadIdx.x]=mask[threadIdx.x] | mask[threadIdx.x+4];
            _CT_mask_dev[th_mappedPos_stream]=mask[threadIdx.x] & _CT_mask_dev[th_mappedPos_stream];   
        }
        mask[threadIdx.x]=0;
    }
}


__global__ void  filterDomainsGPU(unsigned int * _CT_MASKCT_svSize_sval_sSize_sSup_dev, int* _currTable_dev_size, int* _vars_dev, int *_supportOffsetJmp_dev, unsigned int* _supports_dev , int* supportSize_dev){
    

    extern __shared__ unsigned int partialRes[]; //mask (32 ints)

    //int th_col=threadIdx.x/32; //from 0..64 to 0..1 (which column do we look at), each block looks at 2 columns, groups of 32 threads will share the same column

    int th_mappedPos_domain_word=blockIdx.x; //it is thPos as if I didn't have to consider the other streams
    

    if(th_mappedPos_domain_word>=(*supportSize_dev)/32+1){
        return;
    }
    if(_vars_dev[blockIdx.x]==0){
        return;
    }

    //for each bit in the word
    for(int i=0; i<32; i++){  
        
        int mask=1<<(31-(i%32));

        partialRes[threadIdx.x]=0;


        //if value in the domain then we intersect (either all thread are here or none is)
        if((_vars_dev[blockIdx.x] & mask)!=0){
        
            //fino qui sono ok
            int index_x_a=blockIdx.x*32+i;

            //if value in the domain then we intersect (either all thread are here or none is)
            // we do a parallel reduction on the ct
            
            for(int ctW=0; ctW<(*_currTable_dev_size+32); ctW=ctW+32){
                
               
                if(ctW+threadIdx.x < *_currTable_dev_size){

                    //if the intersection is not empyt i can't have partial res empty
                    partialRes[threadIdx.x]=partialRes[threadIdx.x] | (_CT_MASKCT_svSize_sval_sSize_sSup_dev[ctW+threadIdx.x] & _supports_dev[index_x_a*(*_currTable_dev_size)+ctW+threadIdx.x]);
                    
                }
                __syncthreads();
                             
            }

            
            __syncthreads();
            //reduction from 32 to 16
            if(threadIdx.x<16){
                partialRes[threadIdx.x]=partialRes[threadIdx.x] | partialRes[threadIdx.x+16];
            }
            __syncthreads();
            //reduction from 16 to 8
            if(threadIdx.x<8){
                partialRes[threadIdx.x]=partialRes[threadIdx.x] | partialRes[threadIdx.x+8];
            }
            __syncthreads();
            //reduction from 8 to 4
            if(threadIdx.x<4){
                partialRes[threadIdx.x]=partialRes[threadIdx.x] | partialRes[threadIdx.x+4];
            }
            __syncthreads();
            //reduction from 4 to 2
            if(threadIdx.x<2){
                partialRes[threadIdx.x]=partialRes[threadIdx.x] | partialRes[threadIdx.x+2];
            }
            __syncthreads();
            //reduction from 2 to 1
            if(threadIdx.x<1){
                partialRes[threadIdx.x]=partialRes[threadIdx.x] | partialRes[threadIdx.x+1];
                //printf("%%%%%% GPU: th: %d, partialRes: %d\n",threadIdx.x,partialRes[threadIdx.x]);
                if(partialRes[threadIdx.x]==0){  //put 1 in the right position to signal "remove from domain"
                    _vars_dev[th_mappedPos_domain_word]= mask | _vars_dev[th_mappedPos_domain_word];
                }else{ //put 0 in the right position to signal "keep in domain"
                    _vars_dev[th_mappedPos_domain_word]= ~mask & _vars_dev[th_mappedPos_domain_word];
                }
            }
            __syncthreads();
        }
    }

}

int bitsFromRight(int n) {
    //assert 
    if(!(n >= 0 && n < 32)){
        printf("%%%%%% Error: bitsFromRight: n must be between 0 and 31 %d \n",n);
    }
    return (1 << (n)) - 1;
   
}
int bitsFromLeft(int n) {

    
    if (n == 0) return 0;    
    if((n <= 0 || n > 32)){
        printf("%%%%%% Error: bitsFromLeft: n must be between 0 and 31, %d\n",n);
    }   
    return ~0 << (32 - n);
}


void varOffsetLimit(int size,int * where) {
    
    for (int i = 0; i < 32; ++i) {
        where[i] = size / 32;              // Divide the size by the no of streams
    }
    int remainder = size % 32;           // Calculate the remainder

    // Distribute the remainder across the first few parts
    for (int i = 0; i < remainder; ++i) {
        where[i]++;
    }


    where[32]=0;
    where[33]=where[0];
    where[34]=where[1]+where[33];
    where[35]=where[2]+where[34];
    where[36]=where[3]+where[35];
    where[37]=where[4]+where[36];
    where[38]=where[5]+where[37];
    where[39]=where[6]+where[38];
    where[40]=where[7]+where[39];
    where[41]=where[8]+where[40];
    where[42]=where[9]+where[41];
    where[43]=where[10]+where[42];
    where[44]=where[11]+where[43];
    where[45]=where[12]+where[44];
    where[46]=where[13]+where[45];
    where[47]=where[14]+where[46];
    where[48]=where[15]+where[47];
    where[49]=where[16]+where[48];
    where[50]=where[17]+where[49];
    where[51]=where[18]+where[50];
    where[52]=where[19]+where[51];
    where[53]=where[20]+where[52];
    where[54]=where[21]+where[53];
    where[55]=where[22]+where[54];
    where[56]=where[23]+where[55];
    where[57]=where[24]+where[56];
    where[58]=where[25]+where[57];
    where[59]=where[26]+where[58];
    where[60]=where[27]+where[59];
    where[61]=where[28]+where[60];
    where[62]=where[29]+where[61];
    where[63]=where[30]+where[62];
}   