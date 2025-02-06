#include "gpu_constriants/smart_table.cuh"
#include "gpu_constriants/table.cuh"
#include "chrono"
SmartTableGPU::SmartTableGPU(vector<var<int>::Ptr> & vars,  vector<std::vector<int>> & tuples, vector<std::vector<int>> & signs) : SmartTable(vars,tuples,signs){
    
    int noTuples=tuples.size();
    noVars=vars.size();
    
    currTableSize=(noTuples/32)+1; 


    // Memory allocation
    cudaMalloc((void**)&_noVars_dev, sizeof(int));
    cudaMalloc((void**)&_CT_MASKCT_svSize_sval_sSize_sSup_dev, sizeof(unsigned int)*(2*currTableSize+2*noVars+2));
    cudaMalloc((void**)&_supports_dev, sizeof(unsigned int)*_supportSize*currTableSize);
    cudaMalloc((void**)&_supportSize_dev, sizeof(int));
    cudaMalloc((void**)&_variablesOffsets_dev, sizeof(int)*noVars);
    cudaMalloc((void**)&_supportOffsetJmp_dev, sizeof(int)*(noVars+1));
    cudaMalloc((void**)&_currTable_size_dev, sizeof(int));
    cudaMalloc((void**)&_vars_dev, sizeof(int)*((_supportSize/32)+1)); //matrix
    
    cudaMalloc((void**)&workerOffestAndLimit_dev, sizeof(int)*64*noVars);
    
    cudaMalloc((void**)&_tmpMasks, sizeof(unsigned int)*currTableSize*noVars);
    

    //on host side we create simpler structures to then copy the data

    cudaMallocHost((void**)&_CT_MASKCT_svSize_sval_sSize_sSup_host, sizeof(unsigned int)*2*(noVars+1+currTableSize));
    cudaMallocHost((void**)&_vars_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&_vars_to_remove_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&workerOffestAndLimit_host,sizeof(int)*64*noVars);


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

    //compute once and transfer the offsets for the streams:
    noBlocks=(currTableSize);
    noBlocksFilter=((_supportSize/32)+1);

    cudaStreamSynchronize(streams[0]);
 
}

void SmartTableGPU::post(){
    propagate();
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void SmartTableGPU::propagate(){
    //auto t0 = std::chrono::high_resolution_clock::now();
    enfoceGAC();
    //auto t1 = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    //printf("%%%%%% Time taken enfGAC gpu: %ld microseconds\n", duration.count());
    //fflush(stdout);
}

void SmartTableGPU::enfGACDev(){



    for(int i=0;i<currTableSize;i++){
        _CT_MASKCT_svSize_sval_sSize_sSup_host[i]=_currTable._words[i].value();
    }
    
    //aggiungi pure ct ua
    cudaMemcpyAsync(_CT_MASKCT_svSize_sval_sSize_sSup_dev, _CT_MASKCT_svSize_sval_sSize_sSup_host, sizeof(unsigned int)*(2*currTableSize+_s_val.size()+_s_sup.size()+2), cudaMemcpyHostToDevice,streams[0]);   
    
    //metti dump domini qui

    dumpDomainsGPU2();

    cudaMemcpyAsync(_vars_dev, _vars_host, sizeof(int)*((_supportSize/32)+1), cudaMemcpyHostToDevice,streams[0]);

    
    dim3 gridDim(noBlocks,_s_val.size());
    updateTableGPU<<<gridDim,32,32*sizeof(unsigned int),streams[0]>>>(_supports_dev,_CT_MASKCT_svSize_sval_sSize_sSup_dev+(2*currTableSize),_supportOffsetJmp_dev,_CT_MASKCT_svSize_sval_sSize_sSup_dev,_currTable_size_dev,_vars_dev,workerOffestAndLimit_dev,_tmpMasks);          

 

    //we then reduce the matrix of temporary maks into a single mask to add tot he table
    reduce<<<currTableSize,std::min((int)_s_val.size(),32),32*sizeof(int),streams[0]>>>(_CT_MASKCT_svSize_sval_sSize_sSup_dev,_tmpMasks,_currTable_size_dev);
    


    cudaStreamSynchronize(streams[0]);
    
    cudaMemcpyAsync(_CT_MASKCT_svSize_sval_sSize_sSup_host, _CT_MASKCT_svSize_sval_sSize_sSup_dev, currTableSize*sizeof(unsigned int), cudaMemcpyDeviceToHost,streams[0]);    

    //we need to update the current table
    cudaStreamSynchronize(streams[0]);

    _currTable.addToMaskArray(_CT_MASKCT_svSize_sval_sSize_sSup_host);
    
    _currTable.intersectWithMask();
    _currTable.clearMask();

    if(_currTable.isEmpty()){
        //sync stream 0
        failNow();
    }
    filterDomains();


}
void SmartTableGPU::enfoceGAC(){


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
            //overallSize=overallSize+_vars[i]->intialSize();
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

    enfGACDev();
        
}

void SmartTableGPU::dumpDomainsGPU2(){

    //auto start = std::chrono::high_resolution_clock::now();
    for(int index=0; index < noVars; index++){
        //quali variaibli skip

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

    //auto end = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //printf("%%%%%% Time of which by dump 2: %ld microseconds\n", duration.count());
    //fflush(stdout);
    
}