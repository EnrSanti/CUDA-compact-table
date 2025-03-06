#include "gpu_constriants/table.cuh"
#include <chrono>
#include <cuda_runtime.h>
//#define RECORD_OUTPUT

TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){

    #ifdef RECORD_OUTPUT
        auto start= std::chrono::high_resolution_clock::now();
    #endif

    int noTuples=tuples.size();
    noVars=vars.size();
    currTableSize=(noTuples/32)+1; 
    
    // Memory allocation:
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
    //an array containing, for each varaible (depending on the domain size) the amount of work each thread would do in updating the table
    cudaMalloc((void**)&th_limits_dev, sizeof(int)*64*noVars);
    //an list (size no. vars) of arrays of the same size as the CT. it will contain, for each var changed, the mask to add (bitwise AND) to the CT at the end of the update process 
    cudaMalloc((void**)&_tmpMasks, sizeof(unsigned int)*currTableSize*noVars);

    
    
    //on host side we create simpler structures to then copy the data
    cudaMallocHost((void**)&_CT_mask_svs_host, sizeof(unsigned int)*2*(noVars+1+currTableSize));
    cudaMallocHost((void**)&_vars_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&_vars_to_remove_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&th_limits_host,sizeof(int)*64*noVars);


    streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*noStreams);
    cudaError_t err = cudaStreamCreate(&streams[0]);
    
    
    //calculating the amount of rows, for each variable, each thread would have to check
    for(int i=0;i<noVars-1;i++){
        varOffsetLimit(_supportOffsetJmp[i+1]-_supportOffsetJmp[i],th_limits_host+(i*64));
    }
    varOffsetLimit(_supportSize-_supportOffsetJmp[noVars-1],th_limits_host+((noVars-1)*64));


    //copying the data ont he device
    cudaMemcpyAsync(th_limits_dev, th_limits_host, sizeof(int)*64*noVars, cudaMemcpyHostToDevice,streams[0]);
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

    noBlocksFilter=((_supportSize/32)+1);
    
    #ifdef RECORD_OUTPUT
       auto end= std::chrono::high_resolution_clock::now();
       auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
       printf("%%%%%% Time to init table (CUDA): %ld us\n",duration.count());
    #endif
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
    //resetting the vectors
    _s_val.clear(); 
    _s_val.shrink_to_fit();
    _s_sup.clear();
    _s_sup.shrink_to_fit();

    //calculating where the two vectors (sval, ssup will start)
    int internalIndex=currTableSize*2+2;
    
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
        auto start = std::chrono::high_resolution_clock::now();
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
        auto end = std::chrono::high_resolution_clock::now();
        //get the time in micro seconds
        auto start_update=std::chrono::high_resolution_clock::now();
    #endif
    

    //pass: the supports, the changed variables + how many, the indexes for the support, the table and the size, the domains,  and 32*vars ints which tells what range of the varialbe to check according to the index of the th (modifies CT with CT & mask)
    //now each block deals with one specific ct word and also a single changed variable, there can be many blocks, though, even trying to cut them by doing more work per block doesn't improve times
    dim3 gridDim(currTableSize,_s_val.size());
    //pass: the supports, the changed variables + how many, the indexes for the support, the table and the size, the domains,  and 32*vars ints which tells what range of the varialbe to check according to the index of the th (modifies CT with CT & mask)
    updateTableGPU<<<gridDim,32,32*sizeof(unsigned int),streams[0]>>>(_supports_dev,_CT_mask_svs_dev+(2*currTableSize),_supportOffsetJmp_dev,_CT_mask_svs_dev,_currTable_size_dev,_vars_dev,th_limits_dev,_tmpMasks);          
    //updateTableGPU modifies _tmpMasks in global memory, the mask to add to the CT, then reduce<<<>>> does a bit-wise and among the necessary tmpMasks

    //we then reduce the matrix of temporary maks into a single mask to add tot he table
    reduce<<<currTableSize,std::min((int)_s_val.size(),32),32*sizeof(int),streams[0]>>>(_CT_mask_svs_dev,_tmpMasks,_currTable_size_dev);
    
    #ifdef RECORD_OUTPUT
        //syncToRemove
        cudaStreamSynchronize(streams[0]);
        auto end_update=std::chrono::high_resolution_clock::now();
    #endif

    
    //copy back the mask to inteserct with the table calculated by the kernel
    cudaMemcpyAsync(_CT_mask_svs_host, _CT_mask_svs_dev, currTableSize*sizeof(unsigned int), cudaMemcpyDeviceToHost,streams[0]);
    
    cudaStreamSynchronize(streams[0]);

    //adding the retrieved mask
    _currTable.addToMaskArray(_CT_mask_svs_host);
    _currTable.intersectWithMask();
    _currTable.clearMask();

    if(_currTable.isEmpty()){
        //sync stream 0
        failNow();
    }

    //the filtering of the domains is now done host side
    filterDomains();    
    
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
    for(int i=0; i < _s_val.size(); i++){
        
        int index=_s_val[i];
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


//32 threads will do a parallel reduction on the word considered (groups of 32 threads share the same position of the CT)
__global__ void updateTableGPU(unsigned int* _supports_dev,unsigned int * _svSize_off_sval_dev, int *_supportOffsetJmp_dev, unsigned int * _CT_mask_dev,int* _currTable_dev_size, int* _vars_dev, int* offsetsAndLimits,unsigned int* outputMasks){


    extern __shared__ unsigned int mask[]; //mask (32 ints)

    //each thread clears the mask, MANDATORY
    mask[threadIdx.x]=0;

    //get the variable index to consider (we have a grid in which each row considers a different variable )
    int varIndex=_svSize_off_sval_dev[blockIdx.y+2];

    //the starting point (row) of the supports for the var
    int from=_supportOffsetJmp_dev[varIndex];

    //the index of the array offsetsAndLimits which tells the thread how many iterations it will do over the current variable
    int iterations32_th=varIndex*64+threadIdx.x; 

    //the offset of the supports for the current variable for the current thread
    int offset32_th=offsetsAndLimits[iterations32_th+32]; 
    
    //store the number of iterations and the ct_size in registers
    int iterations=offsetsAndLimits[iterations32_th];
    int ct_size=*_currTable_dev_size;
    //each thread does the same (+/-1) amount of iterations,
    for(int j=0; j<iterations; j++){
        
        int wordIndex=(from+j+offset32_th)/32; //piece of row of supports, not the cell, the row piece of row the block looks at

        unsigned int maskContains=1<<(31-j-from-offset32_th+wordIndex*32); //int containing a single bit set

        //check if the value is in the domain, without an if statement
        int condition=((_vars_dev[wordIndex] & maskContains)!=0);
        //calculate the offset of the support word the thread has to (potentially) add to the mask
        int off=(j+offset32_th)*(ct_size)+(from*ct_size)+blockIdx.x; 
        //printf("%%%%%% th. %d, block %d, varIndex %d, condition %d, ctW %d, row %d, specific cell %d \n",threadIdx.x,blockIdx.x,varIndex,condition,blockIdx.x,(j+offset32_th)*(*_currTable_dev_size)+(_supportOffsetJmp_dev[varIndex]*(*_currTable_dev_size)), off);
        
        int support=__ldlu(_supports_dev+off); //load the support word without caching
        mask[threadIdx.x]=mask[threadIdx.x] | (support*condition);           
    }

    //parallel reduction over the 32 threads, each thread took care of a different part of the domain of the same variable
    unsigned result = __reduce_or_sync(0xFFFFFFFF, mask[threadIdx.x]);
    


    //write back the result
    if(threadIdx.x==0){
        outputMasks[varIndex*ct_size+blockIdx.x]=result;
    }

    
}
__global__ void reduce(unsigned int * _CT_mask_svs_dev,unsigned int* tmpMasks, int *ctSize){

    extern __shared__ unsigned int toReduce[]; //mask (32 ints)
    
    int thId=threadIdx.x;
    int ctWord=blockIdx.x;
    int ct_size=*ctSize;
    //get the overall number of vars changed (i.e. how many rows, of a single word do we need to consider)
    int noVarsChanged=_CT_mask_svs_dev[2*ct_size];  

    //how many iterations each of 32 threads need to do
    int iterations=noVarsChanged/32;
    //the number of iterations may not be multiple of 32, so a first portion of the block may need to do +1 iteration
    int lastIteration=noVarsChanged%32;
    //set all bits of the "toReduce" mask to 1, they will be removed via bitwise AND
    toReduce[thId]=0xFFFFFFFF;

    //auxiliary index used for the first part of the block doing an additional iteration 
    int skip=0;
    //each thread will do the same no. of iterations
    for(int i=0;i<iterations;i++){
        //retrieve the variable index to consider (it depends on the thread in the block)
        int varIndex=_CT_mask_svs_dev[2*(ct_size)+2+i*32+threadIdx.x];
        //add (bitwise AND) the proper part of the mask to the final mask 
        toReduce[thId]=toReduce[thId] & tmpMasks[varIndex*(ct_size)+blockIdx.x];
        skip++;
    }
    
    //checking which first portion of the block may need to do an extra iteration 
    if(threadIdx.x<lastIteration){
    
        int varIndex=_CT_mask_svs_dev[2*(ct_size)+2+skip*32+threadIdx.x];
        toReduce[thId]=toReduce[thId] & tmpMasks[varIndex*(ct_size)+blockIdx.x];
    }


    unsigned result = __reduce_and_sync(0xFFFFFFFF, toReduce[threadIdx.x]);
    
    //the first thread of the block intersect the final mask witht the CT and writes the result in global memory
    if(threadIdx.x==0){
        _CT_mask_svs_dev[ctWord]=result&_CT_mask_svs_dev[ctWord];
    }
}



int bitsFromRight(int n) {
    return (1 << (n)) - 1;
}
int bitsFromLeft(int n) {   
    if (n == 0) return 0;    
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
