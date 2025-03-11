#include "gpu_constriants/table.cuh"
#include <chrono>
#include <cuda_runtime.h>
//#define RECORD_OUTPUT
TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){
    
    //get the intial time
    #ifdef RECORD_OUTPUT
        auto start= std::chrono::high_resolution_clock::now();
    #endif

    int noTuples=tuples.size();
    noVars=vars.size();
    currTableSize=(noTuples/32)+1;  
  

    // Memory allocation
    cudaMalloc((void**)&_noVars_dev, sizeof(int)); //the number of variables in the table
    //an array containing the CT (bitMap), a mask, an int containing the size of _s_val, one for the size of _s_sup and then the arrays s_val and s_sup
    cudaMalloc((void**)&_CT_mask_svs_dev, sizeof(unsigned int)*(2*currTableSize+2*noVars+2)); 
    //the support table (copied just once)
    cudaMalloc((void**)&_supports_dev, sizeof(unsigned int)*_supportSize*currTableSize);
    //the size of the support table
    cudaMalloc((void**)&_supportSize_dev, sizeof(int));
    //an array containing the offset (intial value) for each variable, i.e. var 30..50 v1; will contain 30
    cudaMalloc((void**)&_variablesOffsets_dev, sizeof(int)*noVars);
    //an array containing the offset of the supports for each variable
    cudaMalloc((void**)&_supportOffsetJmp_dev, sizeof(int)*(noVars+1));
    //the size (words number, 32 bits) of the current table
    cudaMalloc((void**)&_currTable_size_dev, sizeof(int));
    //an array containing the domains of the variables on the device
    cudaMalloc((void**)&_vars_dev, sizeof(int)*((_supportSize/32)+1)); //matrix
    //an list (size no. vars) of arrays of the same size as the CT. it will contain, for each var changed, the mask to add (bitwise AND) to the CT at the end of the update process 
    cudaMalloc((void**)&_tmpMasks, sizeof(unsigned int)*currTableSize*noVars);
    
    
    //on host side we create simpler structures to then copy the data
    cudaMallocHost((void**)&_CT_mask_svs_host, sizeof(unsigned int)*2*(noVars+1+currTableSize));
    cudaMallocHost((void**)&_vars_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&_vars_to_remove_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix


    streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*1);
    cudaError_t err = cudaStreamCreate(&streams[0]);
    
    
    //copying the data ont he device
    cudaMemcpyAsync(_noVars_dev, &noVars, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supports_dev, _supports, sizeof(unsigned int)*_supportSize*currTableSize, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supportSize_dev, &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_variablesOffsets_dev, _variablesOffsets.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supportOffsetJmp_dev, _supportOffsetJmp.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(&_supportOffsetJmp_dev[noVars], &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_currTable_size_dev, &currTableSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);


    //allocating a buffer of size the maximum number of words the domain of a variable could use 
    int buffSize=0;   
    for(int i=0;i<noVars;i++){
        buffSize=max(buffSize,vars[i]->size()/32+2);
    }
    buffer=(unsigned int*)calloc(buffSize,sizeof(unsigned int));

    //we will launch as many blocks for the filtering process as the overall sum of the domains of the variables 
    noBlocksFilter=((_supportSize/32)+1);


    #ifdef RECORD_OUTPUT
       auto end= std::chrono::high_resolution_clock::now();
       auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
       printf("%%%%%% Time to init table (CUDA): %ld us\n",duration.count());
    #endif

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  
    int maxSharedPerSM=prop.sharedMemPerMultiprocessor;

    if((currTableSize+64)*sizeof(unsigned int)<maxSharedPerSM/4){
        filteringKernel=filterDomainsGPU;
        sharedMemSize=(currTableSize+64);
    }else{
        filteringKernel=filterDomainsGPU2048;
        sharedMemSize=(2048+64);
    }
    
}
void TableGPU::post(){
    propagate();
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void TableGPU::propagate(){
    
    #ifdef RECORD_OUTPUT
        auto start_overall = std::chrono::high_resolution_clock::now();
    #endif

    int internalIndex=currTableSize*2+2;
    _s_val.clear(); 
    _s_val.shrink_to_fit();
    _s_sup.clear();
    _s_sup.shrink_to_fit();
    
    //populate the vector and the respective array counterpart (update s_val)
    for (int i = 0; i < _vars.size(); i++){
        if(_vars[i]->changed()){
            _s_val.push_back(i);
            _CT_mask_svs_host[internalIndex]=i;
            internalIndex++;
        }
    }
   
    //populate the vector and the respective array counterpart (update s_sup)
    for (int i = 0; i < _vars.size(); i++){
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
            _CT_mask_svs_host[internalIndex]=i;
            internalIndex++;
        }
    }


    //add the sizes of the vectors
    _CT_mask_svs_host[currTableSize*2]=_s_val.size();
    _CT_mask_svs_host[currTableSize*2+1]=_s_sup.size();


    #ifdef RECORD_OUTPUT
        auto start_update=std::chrono::high_resolution_clock::now();
    #endif
    
    updateTable();

    //get the time in micro seconds
    #ifdef RECORD_OUTPUT
        auto end_update=std::chrono::high_resolution_clock::now();
    #endif
        
    
    
    //get the current table, from the sparse Bitset to the array
    for(int i=0;i<currTableSize;i++){
        _CT_mask_svs_host[i]=_currTable._words[i].value();
    }
    
    //copy the data on the device
    cudaMemcpyAsync(_CT_mask_svs_dev, _CT_mask_svs_host, sizeof(unsigned int)*(2*currTableSize+_s_val.size()+_s_sup.size()+2), cudaMemcpyHostToDevice,streams[0]);   
    
    //getting the updated domains for the varialbes and copying them on the device
    dumpDomainsGPU2();
    cudaMemcpyAsync(_vars_dev, _vars_host, sizeof(int)*((_supportSize/32)+1), cudaMemcpyHostToDevice,streams[0]);
    #ifdef RECORD_OUTPUT
        auto start_overall_filter = std::chrono::high_resolution_clock::now();
    #endif

    filteringKernel<<<noBlocksFilter,32,sharedMemSize*sizeof(unsigned int)>>>(_CT_mask_svs_dev,_currTable_size_dev,_vars_dev,_supportOffsetJmp_dev,_supports_dev, _supportSize_dev);    
    
    
    #ifdef RECORD_OUTPUT
        cudaStreamSynchronize(streams[0]);
        auto end_overall_filter = std::chrono::high_resolution_clock::now();
    #endif
    //get the time in micro seconds
  
    //getting back the for each varaible the values to remove from the domains
    cudaMemcpyAsync(_vars_to_remove_host, _vars_dev, sizeof(int)*((_supportSize/32)+1), cudaMemcpyDeviceToHost,streams[0]);

    //adding the retrieved mask
    _currTable.addToMaskArray(_CT_mask_svs_host);
    _currTable.intersectWithMask();
    _currTable.clearMask();

    if(_currTable.isEmpty()){
        failNow();
    }

 


    //wait for  the domains to be copied back
    cudaStreamSynchronize(streams[0]);
    
    #ifdef RECORD_OUTPUT
        auto start_removing = std::chrono::high_resolution_clock::now(); 
    #endif
    
    //for all the vars in ssup, update their domains with the information from the last kernel by checking _vars_to_remove_host
    for(int i=0;i<_s_sup.size();i++){

        int index=_CT_mask_svs_host[i+currTableSize*2+_s_val.size()+2];
        int starting_value=(_supportOffsetJmp[index]-_variablesOffsets[index]+_vars[index]->min());
        int starting_word=(starting_value)/32;
        int starting_bit=(starting_value)%32; 
        
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

    
    #ifdef RECORD_OUTPUT
        auto end_overall = std::chrono::high_resolution_clock::now();
        auto duration_removing = std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_removing);
        auto duration_overall_filter = std::chrono::duration_cast<std::chrono::microseconds>(end_overall_filter - start_overall_filter);
        auto duration_dump_cpu = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        auto duration_update = std::chrono::duration_cast<std::chrono::microseconds>(end_update - start_update);

        auto duration_overall = std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall);
        printf("%%%%%% Time to propagate (CUDA): %ld us (to dump & cpy %ld) (update %ld) (filter %ld) (removing %ld)\n",duration_overall.count(),duration_dump_cpu.count(),duration_update.count(),duration_overall_filter.count(),duration_removing.count());
    #endif    
}

void TableGPU::dumpDomainsGPU2(){

    //for each of the vars, dump the domain which is kept as a sparseBitset into the array (treat it as a black box)
    for(int i=0; i < _s_sup.size(); i++){
        int index=_s_sup[i];
        //which vars i don't need to update
        if(!(_vars[index]->changed()) && _vars[index]->size()==1)
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
    }

}


__global__ void  filterDomainsGPU(unsigned int * _CT_mask_svs_dev, int* _currTable_dev_size, int* _vars_dev, int *_supportOffsetJmp_dev, unsigned int* _supports_dev , int* supportSize_dev){
    

    extern __shared__ unsigned int partialRes[]; //mask (64 ints, the last 32 int which is used to store _vars_dev[blockIdx.x], so it's accessed just once)

    //do it 32 times, so no ifs
    partialRes[threadIdx.x+32]=_vars_dev[blockIdx.x]; //store the word of the domain in the last int of the shared memory
    
    //if the word is empty, no need to do anything, i can't remove any values
    if(partialRes[32]==0){
        return;
    }

    int ct_size=*_currTable_dev_size;

    int skip=0;
    for(int ctW=0; ctW<=(ct_size)-32; ctW=ctW+32){
        partialRes[ctW+threadIdx.x+64]=__ldlu(_CT_mask_svs_dev+ctW+threadIdx.x);
        skip+=32;
    }
 
    //unroll of the last iteration of the loop, if the CT size is not a multiple of 32:
    if(threadIdx.x<(*_currTable_dev_size)%32){
        partialRes[skip+threadIdx.x+64]=__ldlu(_CT_mask_svs_dev+skip+threadIdx.x);
    }
    __syncthreads();

    //for each bit in the word, each thread in the block will do 32 iterations
    for(int i=0; i<32; i++){  
        
        int mask=1<<(31-i);
        partialRes[threadIdx.x]=0;
        
        //if value in the domain then we intersect (either all thread are here or none is)
        if((partialRes[32] & mask)!=0){
            
            
            //the index of the support i look at, it doesn't depend on the thread just on the block, each thread will do a different piece of work on the CT
            int index_x_a=blockIdx.x*32+i;
            int index_ctSize=index_x_a*(ct_size);
            
            //the parallel reduction part
            skip=0;
            for(int ctW=0; ctW<=(ct_size)-32; ctW=ctW+32){
                //we add to the partial result
                //printf("%%%%%% INSIDE LOOP %d %d %d\n",index_x_a,ctW,threadIdx.x);
                int support=__ldlu(_supports_dev+ index_ctSize+ctW+threadIdx.x); //load the support word without caching
                partialRes[threadIdx.x]=partialRes[threadIdx.x] | (partialRes[ctW+threadIdx.x+64] & support);
                //increment by 32 for the last (unrolled iterations, look after the for loop)
                skip+=32;
            }

            //unroll of the last iteration of the loop, if the CT size is not a multiple of 32 then if(threadIdx.x<=(*_currTable_dev_size)%32) the threads considered will do one more iteration
            int condition=(threadIdx.x<(ct_size)%32);
            //printf("%%%%%% OUTSIDE LOOP, th %d condition %d\n",threadIdx.x,condition);

            partialRes[threadIdx.x]=partialRes[threadIdx.x] | ( (condition) * (partialRes[skip+threadIdx.x+64] & _supports_dev[index_ctSize+skip+threadIdx.x]));
            //end of unrolled loop

            //reduction among the 32 threads of the block // e questa è ok
            unsigned result = __reduce_or_sync(0xFFFFFFFF, partialRes[threadIdx.x]);
        
            
            //if a bit is set to 1 then the value specified by the position needs to be removed from the domain, otherwise not
            //it's just a "complex" operation to avoid an if statement
            partialRes[threadIdx.x+32] = (partialRes[32] & ~(1 << (31-i))) | ((result == 0) << (31-i));
            //in the previous instruction we are only insterested in partialRes[33] but instead of doing an if and having a sync, we do 31 useless operation on the other threads (1 per thread)           
        }
    }
    if(threadIdx.x==0){
        //write back the result, one access in global memory
        _vars_dev[blockIdx.x]=partialRes[32];
    }   
}


__global__ void  filterDomainsGPU2048(unsigned int * _CT_mask_svs_dev, int* _currTable_dev_size, int* _vars_dev, int *_supportOffsetJmp_dev, unsigned int* _supports_dev , int* supportSize_dev){
    

    extern __shared__ unsigned int partialRes[]; //mask (64 ints, the last 32 int which is used to store _vars_dev[blockIdx.x], so it's accessed just once)

    //do it 32 times, so no ifs
    partialRes[threadIdx.x+32]=_vars_dev[blockIdx.x]; //store the word of the domain in the last int of the shared memory
    
    //if the word is empty, no need to do anything, i can't remove any values
    if(partialRes[32]==0){
        return;
    }
    int ct_size=*_currTable_dev_size;

    bool sharedOver=false;
    for(int ctW=0; ctW<2048; ctW=ctW+32){
        partialRes[ctW+threadIdx.x+64]=__ldlu(_CT_mask_svs_dev+ctW+threadIdx.x);
    }
 

    //for each bit in the word, each thread in the block will do 32 iterations
    for(int i=0; i<32; i++){  
        
        int mask=1<<(31-i);
        partialRes[threadIdx.x]=0;
        
        //if value in the domain then we intersect (either all thread are here or none is)
        if((partialRes[32] & mask)!=0){
            
            
            //the index of the support i look at, it doesn't depend on the thread just on the block, each thread will do a different piece of work on the CT
            int index_x_a=blockIdx.x*32+i;
            int index_ctSize=index_x_a*(ct_size);
            
            //the parallel reduction part
            int skip=0;
            for(int ctW=0; ctW<=(ct_size)-32; ctW=ctW+32){
                //we add to the partial result
                //printf("%%%%%% INSIDE LOOP %d %d %d\n",index_x_a,ctW,threadIdx.x);

                int support=__ldlu(_supports_dev+ index_ctSize+ctW+threadIdx.x); //load the support word without caching
                if(ctW+threadIdx.x<2048){
                    partialRes[threadIdx.x]=partialRes[threadIdx.x] | (partialRes[ctW+threadIdx.x+64] & support);    
                }else{
                    partialRes[threadIdx.x]=partialRes[threadIdx.x] | (_CT_mask_svs_dev[ctW+threadIdx.x]  & support);
                }
                //increment by 32 for the last (unrolled iterations, look after the for loop)

                skip+=32;
            }

            //unroll of the last iteration of the loop, if the CT size is not a multiple of 32 then if(threadIdx.x<=(*_currTable_dev_size)%32) the threads considered will do one more iteration
            int condition=(threadIdx.x<(ct_size)%32);
            //printf("%%%%%% OUTSIDE LOOP, th %d condition %d\n",threadIdx.x,condition);
            partialRes[threadIdx.x]=partialRes[threadIdx.x] | ( (condition) * (_CT_mask_svs_dev[skip+threadIdx.x]  & _supports_dev[index_ctSize+skip+threadIdx.x]));
           
            //end of unrolled loop

            //reduction among the 32 threads of the block // e questa è ok
            unsigned result = __reduce_or_sync(0xFFFFFFFF, partialRes[threadIdx.x]);
        
            
            //if a bit is set to 1 then the value specified by the position needs to be removed from the domain, otherwise not
            //it's just a "complex" operation to avoid an if statement
            partialRes[threadIdx.x+32] = (partialRes[32] & ~(1 << (31-i))) | ((result == 0) << (31-i));
            //in the previous instruction we are only insterested in partialRes[33] but instead of doing an if and having a sync, we do 31 useless operation on the other threads (1 per thread)           
        }
    }
    if(threadIdx.x==0){
        //write back the result, one access in global memory
        _vars_dev[blockIdx.x]=partialRes[32];
    }   
}


int bitsFromRight(int n) {
    return (1 << (n)) - 1;
}
int bitsFromLeft(int n) {   
    if (n == 0) return 0;    
    return ~0 << (32 - n);
}
