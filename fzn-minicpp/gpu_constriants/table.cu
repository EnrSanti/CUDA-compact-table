#include "gpu_constriants/table.cuh"

TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){
    setPriority(CLOW);
    
    printf("%%%%%% TableGPU constructor\n");
    //get the number of tuples and vars in the table
    int noTuples=tuples.size();
    noVars=vars.size();
    
    //the number of words used to store the table
    currTableSize=(noTuples/32)+1;

    //pre calculate the number of blocks to launch    
    int noBlocks=(currTableSize/32)+1;

    //allocating & copying the number of vars to the dev
    noVars_dev=mallocDevice<int>(sizeof(int));
    cudaMemcpyAsync(noVars_dev, &noVars, sizeof(int), cudaMemcpyHostToDevice);
    
    //device stuff
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    sm_count = device_prop.multiProcessorCount;
    int cores_per_SM = 128;

    //printf("%%%%%% number of SMs: %d\n",sm_count);
    //printf("%%%%%% warp size: %d\n",32);
    //printf("%%%%%% cores per SM: %d\n",cores_per_SM);
    //printf("%%%%%% support size: %d\n",_supportSize);
    
    
    // Memory allocation for the device
    currTable_dev = mallocDevice<unsigned int>(sizeof(unsigned int)*currTableSize); 
    currTable_mask_dev = mallocDevice<unsigned int>(sizeof(unsigned int)*currTableSize); 
    supports_dev = mallocDevice<unsigned int>(sizeof(unsigned int)*_supportSize*currTableSize);
    supportSize_dev = mallocDevice<int>(sizeof(int));
    variablesOffsets_dev = mallocDevice<int>(sizeof(int)*noVars);
    supportOffsetJmp_dev = mallocDevice<int>(sizeof(int)*(noVars+1));
    currTable_size_dev=mallocDevice<int>(sizeof(int));
    s_val_size_dev=mallocDevice<int>(sizeof(int));
    offset=mallocDevice<int>(sizeof(int));
    s_val_dev=mallocDevice<int>(sizeof(int)*noVars);
    vars_dev=mallocDevice<unsigned int>(sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    output_dev=mallocDevice<int>(sizeof(int)*(currTableSize/32)+1); //one for each block


    //on host side we create simpler structures to then copy the data
    currTable_host=mallocHost<unsigned int>(sizeof(unsigned int)*currTableSize); 
    unsigned int *_supports_host = mallocHost<unsigned int>(sizeof(unsigned int)*_supportSize*currTableSize);
    vars_host=mallocHost<int>(sizeof(unsigned int)*((_supportSize/32)+1)); //matrix
    outputArray=mallocHost<int>(sizeof(int)*(currTableSize/32)+1); 

    //get the vectors to arrays (not the best but ok)
    for(int i=0;i<_supportSize;i++){
        for(int j=0; j<currTableSize;j++){
            _supports_host[i*currTableSize+j]=_supports[i]._words[j].value();
        }
        
    }

    //initialize the vars_host
    for(int i=0;i<((_supportSize/32)+1);i++){
        vars_host[i]=0;
    }

    //can be done much better but for now it's ok (done once)
    for(int i=0;i<noVars;i++){
        vector<int> dom=_vars[i]->dumpDomainToVec();
        for(int j=0;j<dom.size();j++){
            //getting an unsigned int with the 32-dom[j]-_variablesOffsets[i] bit set
            unsigned int mask=1<<31-(dom[j]-_variablesOffsets[i]+_supportOffsetJmp[i]);
            int starting_word=(dom[j]-_variablesOffsets[i]+_supportOffsetJmp[i])/32;
            vars_host[starting_word]=vars_host[starting_word]|mask;
            //prinitng bits of _vars_host
        }
    }

            
    //end of could be done better
    *currTable_host=_currTable._words.data()->value();
    


    //Memory copy all data to the dev
    cudaMemcpyAsync(supports_dev, _supports_host, sizeof(unsigned int)*_supportSize*currTableSize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(currTable_dev, currTable_host, sizeof(unsigned int)*currTableSize, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(supportSize_dev, &_supportSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(variablesOffsets_dev, _variablesOffsets.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(supportOffsetJmp_dev, _supportOffsetJmp.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(&supportOffsetJmp_dev[noVars], &_supportSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(currTable_size_dev, &currTableSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(vars_dev, vars_host, sizeof(unsigned int)*((_supportSize/32)+1), cudaMemcpyHostToDevice);

    cudaFree(_supports_host);
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

    //copy the table from host to device
    cudaMemcpyAsync(currTable_dev, currTable_host, sizeof(unsigned int)*currTableSize, cudaMemcpyHostToDevice);
   
    //reset var_host
    for(int i=0; i<((_supportSize/32)+1);i++){
        vars_host[i]=0;
    }

    //can be done much better but for now it's ok
    for(int i=0;i<noVars;i++){
        vector<int> dom=_vars[i]->dumpDomainToVec();
        //*****se avessi dom come vec potrei copiarlo su gpu e fare tutto li
        for(int j=0;j<dom.size();j++){
            //getting an unsigned int with the 32-dom[j]-_variablesOffsets[i] bit set
            unsigned int mask=1<<31-(dom[j]-_variablesOffsets[i]+_supportOffsetJmp[i]);
            //printing the domain
            int starting_word=(dom[j]-_variablesOffsets[i]+_supportOffsetJmp[i])/32;
            vars_host[starting_word]=vars_host[starting_word]|mask;
        }
    }
    //*****così da rimuovere questo
    cudaMemcpyAsync(vars_dev, vars_host, sizeof(unsigned int)*((_supportSize/32)+1), cudaMemcpyHostToDevice);

    //end of could be done better
    //*****questa verrebbe eseguita in //
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

    cudaMemcpyAsync(s_val_size_dev, &size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(s_val_dev, _s_val.data(), sizeof(int)*size, cudaMemcpyHostToDevice);
    int offset_=0;
    int min=noVars;
    for(int i=0;i<size;i++){
        if(_s_val[i]<min){
            offset_=_s_val[i];
        }
    }
    cudaMemcpyAsync(offset, &offset_, sizeof(int), cudaMemcpyHostToDevice);

 
    for(int i=0;i<currTableSize;i++){
        currTable_host[i]=_currTable._words[i].value();
    }
    cudaMemcpyAsync(currTable_dev,currTable_host, sizeof(unsigned int)*currTableSize, cudaMemcpyHostToDevice);

    updateTableGPU<<<noBlocks,32,32*sizeof(unsigned int)>>>(supports_dev,s_val_size_dev,s_val_dev,supportSize_dev,variablesOffsets_dev,supportOffsetJmp_dev,currTable_dev,currTable_size_dev,vars_dev,output_dev, offset, noVars_dev);

    
    //retrieve the output from the device
    cudaMemcpyAsync(outputArray, output_dev, sizeof(int)*noBlocks, cudaMemcpyDeviceToHost);


    //performed on host, the number of blocks usually is small (e.g. if we have 1280 rows in the table we have 2 blocks)

    for(int i=0; i<noBlocks; i++){
        if(outputArray[i]==1){
            output=1;
            break;
        }
    }
    if(output==1){
        failNow();
    }else{
        //we retrieve current table
        //getting back the current table
        cudaMemcpyAsync(currTable_host, currTable_dev, sizeof(unsigned int)*(currTableSize), cudaMemcpyDeviceToHost);
        //we need to update the current table
        for(int i=0;i<currTableSize;i++){
            _currTable._mask[i]=currTable_host[i];
        }
        _currTable.intersectWithMask();
        _currTable.clearMask();
        if(_currTable.isEmpty()){
            failNow();
        }
    }
    
    
	filterDomains();

    

}
//same as the CPU version   
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
        //_vars[index]->dumpInSparseBitSet(index,_variablesOffsets[index],_vars[index]->min(),_vars[i]->initialMin(),_vars[index]->max(),_lastVarsValues[index]);
    }
}

// 1 th per support row
__global__ void updateTableGPU(unsigned int* _supports_dev,int * _s_val_size_dev, int *_s_val_dev, int *_supportSize_dev, int *_variablesOffsets_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* _currTable_dev_size, unsigned int* _vars_dev, int* output, int *offset, int *varNoDev){


    int thPos = blockIdx.x * blockDim.x + threadIdx.x; //which currTable word we are considering
    int varIndex=0;
    extern __shared__ unsigned int mask[]; //mask (32)
    
    //clear mask MANDATORY

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
/*
void TableGPU::print(){    
    
    printf("%%%%%% ----------------- VARS: -----------------\n\n");
    for (int i = 0; i < _vars.size(); i++){
        printf("%%%%%% Var %d: %d\n",i,_vars[i]->getId());      
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
}*/