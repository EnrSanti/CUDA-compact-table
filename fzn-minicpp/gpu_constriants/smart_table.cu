#include "gpu_constriants/smart_table.cuh"
#include "gpu_constriants/table.cuh"
SmartTableGPU::SmartTableGPU(vector<var<int>::Ptr> & vars,  vector<std::vector<int>> & tuples, vector<std::vector<int>> & signs) : SmartTable(vars,tuples,signs){
   setPriority(CLOW);

    
int noTuples=tuples.size();
    noVars=vars.size();
    
    currTableSize=(noTuples/32)+1; 
    
    

    // Memory allocation
    _noVars_dev=mallocDevice<int>(sizeof(int));
    _CT_MASKCT_svSize_sval_sSize_sSup_dev = mallocDevice<unsigned int >(sizeof(unsigned int)*(2*currTableSize+2*noVars+2)); 
    _supports_dev = mallocDevice<unsigned int>(sizeof(unsigned int)*_supportSize*currTableSize);
    _supportSize_dev = mallocDevice<int>(sizeof(int));
    _variablesOffsets_dev = mallocDevice<int>(sizeof(int)*noVars);
    _supportOffsetJmp_dev = mallocDevice<int>(sizeof(int)*(noVars+1));
    _currTable_size_dev=mallocDevice<int>(sizeof(int));
    _vars_dev=mallocDevice<int>(sizeof(int)*((_supportSize/32)+1)); //matrix
    
    workerOffestAndLimit_dev=mallocDevice<int>(sizeof(int)*64*noVars);
    
    

    //on host side we create simpler structures to then copy the data

    cudaMallocHost((void**)&_CT_MASKCT_svSize_sval_sSize_sSup_host, sizeof(unsigned int)*2*(noVars+1+currTableSize));
    cudaMallocHost((void**)&_vars_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    cudaMallocHost((void**)&_vars_to_remove_host, sizeof(unsigned int)*((_supportSize/32)+1)); //matrix

    cudaMallocHost((void**)&dumped, sizeof(bool)*noVars); 
    //initialize it to false
    for(int i=0;i<noVars;i++){
        dumped[i]=false;
    }

    cudaMallocHost((void**)&workerOffestAndLimit_host,sizeof(int)*64*noVars);


    streams=(cudaStream_t*)malloc(sizeof(cudaStream_t)*noStreams);


    

    cudaError_t err = cudaStreamCreate(&streams[0]);
    
    if (err != cudaSuccess) {
        printf("%%%%%% Error creating stream: %s\n", cudaGetErrorString(err));
    }

    for(int i=0;i<noVars-1;i++){
        varOffsetLimit(_supportOffsetJmp[i+1]-_supportOffsetJmp[i],workerOffestAndLimit_host+(i*64));
    }
    varOffsetLimit(_supportSize-_supportOffsetJmp[noVars-1],workerOffestAndLimit_host+((noVars-1)*64));

    cudaMemcpyAsync(workerOffestAndLimit_dev, workerOffestAndLimit_host, sizeof(int)*64*noVars, cudaMemcpyHostToDevice,streams[0]);



    cudaMemcpyAsync(_noVars_dev, &noVars, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    
    //Memory copy

    cudaMemcpyAsync(_supports_dev, _supports, sizeof(unsigned int)*_supportSize*currTableSize, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supportSize_dev, &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_variablesOffsets_dev, _variablesOffsets.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(_supportOffsetJmp_dev, _supportOffsetJmp.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice,streams[0]);
    cudaMemcpyAsync(&_supportOffsetJmp_dev[noVars], &_supportSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);

    cudaMemcpyAsync(_currTable_size_dev, &currTableSize, sizeof(int), cudaMemcpyHostToDevice,streams[0]);


    //compute once and transfer the offsets for the streams:
    noBlocks=(currTableSize/4)+1;
    noBlocksFilter=((_supportSize/32)+1);

    cudaStreamSynchronize(streams[0]);
 
}

void SmartTableGPU::post(){
    //printf("%%%%%% post GPU\n");
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void SmartTableGPU::propagate(){
    //printf("%%%%%% propagate on GPU\n");
    enfoceGAC();
}

void SmartTableGPU::enfGACDev(){




    int output=0;
    

    for(int i=0;i<currTableSize;i++){
        _CT_MASKCT_svSize_sval_sSize_sSup_host[i]=_currTable._words[i].value();
    }
    
    //aggiungi pure ct ua
    cudaMemcpyAsync(_CT_MASKCT_svSize_sval_sSize_sSup_dev, _CT_MASKCT_svSize_sval_sSize_sSup_host, sizeof(unsigned int)*(2*currTableSize+_s_val.size()+_s_sup.size()+2), cudaMemcpyHostToDevice,streams[0]);   
    
    //metti dump domini qui

    dumpDomainsGPU();

    cudaMemcpyAsync(_vars_dev, _vars_host, sizeof(int)*((_supportSize/32)+1), cudaMemcpyHostToDevice,streams[0]);

    
    /*    
    int offset=0;
    int domainSize=0;

    //for each changed var copy just their domain
    for(int i=0;i<_s_val.size();i++){
        int index=_s_val[i];
        offset=(_supportOffsetJmp[index])/32;
        int words_to_reset=-1;
        int to=-1;
        if(index<noVars-1){
            to=_supportOffsetJmp[index+1]/32;
            domainSize=to-offset+1;
        }else{
            to=(_supportSize/32)+1;
            domainSize=to-offset;
        }

        //cudaMemcpyAsync(_vars_dev, _vars_host, sizeof(unsigned int)*((_supportSize/32)+1), cudaMemcpyHostToDevice,streams[0]);
        cudaMemcpyAsync(_vars_dev+offset, _vars_host+offset, sizeof(unsigned int)*domainSize, cudaMemcpyHostToDevice,streams[0]);
    }
    */
    

    //pass: the supports, the changed variables + how many, the indexes for the support, the table and the size, the domains,  and 32*vars ints which tells what range of the varialbe to check according to the index of the th (modifies CT with CT & mask)
    updateTableGPU<<<noBlocks,128,128*sizeof(unsigned int),streams[0]>>>(_supports_dev,_CT_MASKCT_svSize_sval_sSize_sSup_dev+(2*currTableSize),_supportOffsetJmp_dev,_CT_MASKCT_svSize_sval_sSize_sSup_dev,_currTable_size_dev,_vars_dev,workerOffestAndLimit_dev);          

    
    //si copio mask in ct per semplcità
    cudaMemcpyAsync(_CT_MASKCT_svSize_sval_sSize_sSup_host, _CT_MASKCT_svSize_sval_sSize_sSup_dev, currTableSize*sizeof(unsigned int), cudaMemcpyDeviceToHost,streams[0]);

    filterDomainsGPU<<<noBlocksFilter,32,32*sizeof(unsigned int),streams[0]>>>(_CT_MASKCT_svSize_sval_sSize_sSup_dev,_currTable_size_dev,_vars_dev,_supportOffsetJmp_dev,_supports_dev, _supportSize_dev);
    //launch filtering 

    //each block does 2 words of the domains

   
    //we need to update the current table

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
void SmartTableGPU::enfoceGAC(){
    cudaDeviceSynchronize();
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

    

    //if(overallSize>300){ //to better see advantages when testing remove and do only enfGACDev();
    
    enfGACDev();

  
    
    //}else{
    //    updateTable();
    //}

    //filterDomains();

    
    //printf("%%%%%% ------------------------------------------------------------------ \n");
        
}

void SmartTableGPU::dumpDomainsGPU(){
    
    for(int index=0; index < noVars; index++){
        //quali variaibli skippo

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
        
        for (int j = _vars[index]->min(); j <= _vars[index]->max();  j++){ 

            if(_vars[index]->contains(j)){
                int wordIndex=(j-_variablesOffsets[index]+_supportOffsetJmp[index])/32;
                _vars_host[wordIndex]=_vars_host[wordIndex]|(0x80000000>>(((_supportOffsetJmp[index]+j-_variablesOffsets[index])% 32 + 32)%32));
            }

        }   
        
        dumped[index]=true;

    }

}
