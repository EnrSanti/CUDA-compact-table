#include "table.hpp"

Table::Table(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) :
    Constraint(vars[0]->getSolver()), 
    _vars(vars), _tuples(tuples), 
    _currTable(SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size())){
    
    int noTuples=tuples.size();
    int noVars=vars.size();
    _s_val= vector<int>();
    _s_sup= vector<int>();
    _supportOffsetJmp=vector<int>(noVars);
    _variablesOffsets=vector<int>(noVars);
    _deltaXs=vector<unsigned int*>(noVars,nullptr);
    _lastVarsValues=vector<SparseBitSet>(noVars,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),0));

    for (int i = 0; i < noVars; i++){        
        //calculating the number of rows in the support bitset
        _supportSize+=vars[i]->size();
        //we store the offset
        _variablesOffsets[i]=vars[i]->min();

        //we allocate the delta and lastVarsValues
        _deltaXs[i]=new unsigned int[vars[i]->getSizeOfBitSet()];
        _lastVarsValues[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),_vars[i]->size());

        //initialize lastVarsValues
        vars[i]->dumpInSparseBitSet(vars[i]->min(),vars[i]->max(),_lastVarsValues[i]);

    }

    //calculating the offset of the variables, used in accessing the support rows    
    _supportOffsetJmp[0]=0;
    for (int i = 1; i < noVars; i++){
        _supportOffsetJmp[i]=_supportOffsetJmp[i-1]+vars[i-1]->size();
    }

    //we allocate and initialize the support bitsets
    _supports=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _supportsShort=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _supportsMin=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _supportsMax=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
   

    _residues= vector<trail<int>>(_supportSize);


    /*
    printf("%%%%%% the supports will have size (rows): %d \n",_supportSize);
    for (int i = 0  ; i < noVars; i++){
         printf("%%%%%% the var[%d] starts at: %d \n",i,_supportOffsetJmp[i]);
         printf("%%%%%% the var[%d] has initial size %d \n",i,vars[i]->intialSize());
    }
    */

    //we allocate and initialize the support bitsets
    for (int i = 0; i < _supportSize; i++){
        _supports[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);//the content doesn't make sense yet, later we need to update the mask and intersect it
        _supportsShort[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        _supportsMax[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        _supportsMin[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
    }



    for (int t = 0; t < noTuples; t++){
        for (int v = 0; v < noVars; v++){
            //classical entry (we update all the supports in the same way)
            int entryValue=tuples[t][v]-_variablesOffsets[v];   
            int offset=_supportOffsetJmp[v]+entryValue;

            //TO UNCOMMENT ONCE ALL ENTRIES ARE IMLPMENTED
            //we check if it's not a short entry (i.e. has *)
            /*if(strncmp(var,"*",1)==0){
                //supports, supportMin, supportMax we need to set all the var values to 1
                //done
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);  
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);   
                }
                //supportsShort to 0, so we don't add anything :)

            }else if(strncmp(var,"!=",2)==0){ //if we have a smart entry !=
                //done
                int entryValue=atoi(var+2)-data.variablesOffsets[ctr]; //we skip over the chars != (no spaces allowed), -varOffset, to retrieve the index of the value on it's domain, i.e. x 4..10, if we write 6 we mean the second possible value of x 
                for (int varValue = 0; varValue < entryValue; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);   
                }
                for (int varValue = entryValue+1; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);   
                }
                //TODO CHECK IF THE LAST VALUE HAS TO BE INCLUDED (i.e. x!=c for supportMin or x!=a for supportMax)
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);  
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);   
                }
                // supportsShort to 0, so we don't add anything :)
            
            }else if(strncmp(var,"<=",2)==0){

                //done
                int entryValue=atoi(var+2)-data.variablesOffsets[ctr];

                for (int varValue = 0; varValue <= entryValue; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);
                }

                //set supportsMax all to 1
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);   
                }
                // supportsShort to 0, so we don't add anything :)

            }else if(strncmp(var,"<",1)==0){
                //done
                int entryValue=atoi(var+1)-data.variablesOffsets[ctr];


                for (int varValue = 0; varValue < entryValue; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);
                }
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);   
                }
                // supportsShort to 0, so we don't add anything :)
            }else if(strncmp(var,">=",2)==0){
                //done
                int entryValue=atoi(var+2)-data.variablesOffsets[ctr];

                for (int varValue = entryValue; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);
                }       
                //set supportsMin all to 1
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMax
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);   
                }

                // supportsShort to 0, so we don't add anything :)
            }else if(strncmp(var,">",1)==0){
                int entryValue=atoi(var+1)-data.variablesOffsets[ctr];
                for (int varValue = entryValue+1; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);
                }  
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);   
                }
                // supportsShort to 0, so we don't add anything :)
            }else{*/
           
            _supports[offset].addToMaskInt(t+1); 
            _supportsShort[offset].addToMaskInt(t+1);

            for (int varValue = 0; varValue <= entryValue; varValue++) {
                offset=_supportOffsetJmp[v]+varValue;
                //update supportsMin
                _supportsMin[offset].addToMaskInt(t+1);  
            }
            for (int varValue = entryValue; varValue < vars[v]->intialSize(); varValue++) {
                offset=_supportOffsetJmp[v]+varValue;
                //update supportsMax
                _supportsMax[offset].addToMaskInt(t+1);
            }
        }
    }
    

    int bitsPerWord=32;

    for (int i = 0; i < _supportSize; ++i){  
        _supports[i].intersectWithMask();
        _supportsShort[i].intersectWithMask();
        _supportsMin[i].intersectWithMask();
        _supportsMax[i].intersectWithMask();

        _supports[i].clearMask();
        _supportsShort[i].clearMask();
        _supportsMin[i].clearMask();
        _supportsMax[i].clearMask();
        
        //we initialize residues
        for(int j=0; j<noTuples; j++){
            if(_supports[i]._words[j/bitsPerWord]!=0x00000000){
                _residues[i]=trail<int>(vars[0]->getSolver()->getStateManager(), j); 
            }else{
                _residues[i]=trail<int>(vars[0]->getSolver()->getStateManager(), 0); 
            }
        }
    }
    print();
}

void Table::post()
{
    printf("%%%%%% Table post called.\n");
    for (auto const & v : _vars){
        v->propagateOnDomainChange(this);
    }

}

void Table::propagate()
{
    printf("%%%%%% Table propagation called.\n");
    enfoceGAC();
}



//---------------------------------------------
//------- The three functions of alg. 2 -------
//---------------------------------------------

void Table::updateTable(){
    /*
    for(int i=0; i < _s_val.size(); i++){
        _currTable.clearMask();
        if(){
            //incremental update
        }
    }
	for (int i=0; i<data->variablesNo; i++){
		if(data->s_val[i]==1){
			clearMask(&(data->currTable));

			if(deltaXSizes[i]+2<domainSizes[i] && 1==0){ //TODO TOGLI SECONDA CONDIZIONE, in questa versione incremental update non è considerato
				//printf("\n++++incremental update++++\n");
				int domMin=-1; //TODO CAMBIA
				int domMax=-1; //TODO CAMBIA
				for (int j = 0; j < deltaXSizes[i]; j++){
					if(deltaXs[i][j]>domMin && deltaXs[i][j]<domMax){
						int index=getSupportIndex(data,i,deltaXs[i][j]);
						addToMask(&(data->currTable),data->supportsShort[index].words); 		
					}
				}
				reverseMask(&(data->currTable));
				//TODO UNCOMMENT
				//if(dom(i).minChanged()){
				//	...	
				//}
				//if(dom(i).maxChanged()){
				//	...
				//}
				
            
			}else{
				//printf("\n++++reset based update++++\n"); 
				for (int j = 0; j < data->supportSizes[i]; j++){
					if(domains[i][j]==1){
						int index=getSupportIndex(data,i,j+data->variablesOffsets[i]);
						addToMask(&(data->currTable),data->supports[index].words); 		
					}
				}
			}
			intersectWithMask(&(data->currTable));
			if(isEmpty(data->currTable)){
				return;
			}
		}
	}
    */
}

/*
void Table::filterDomains(CT *data,char** domains,int* domainSizes){
	//for now we loop through all the array, with a list we could be more efficient (todo)
	for (int i = 0; i < data->variablesNo; i++){
		if(data->s_sup[i]==1){
			//for all values in dom x, che si possono prendere dal processo di search (for now we loop through all the array todo: lists)
			for (int j = 0; j < data->supportSizes[i]; j++){
				if(domains[i][j]==1){ //i.e. a \in dom(x)
					int x_aIndex=getSupportIndex(data,i,j+data->variablesOffsets[i]);
					int index=data->residues[x_aIndex];
					if((data->currTable.words[index] & (data->supports[x_aIndex]).words[index] ) == 0x0000000000000000){
						index=intersectIndex(&(data->currTable),data->supports[x_aIndex].words);
						if(index!=-1){
							data->residues[x_aIndex]=index;
						}else{
							if(domains[i][j]!=0)
								domainSizes[i]-=1;
							domains[i][j]=0;

						}

					}
				}
			}
			data->lastSizes[i]=domainSizes[i];//=dom(x) todo;
		}
	}
}
*/

void Table::enfoceGAC(){

    //printing the size
    //printf("%%%%%% %ld\n",_vars.size());
    //update the table
	
    //update the table
    _s_val.clear();
    _s_sup.clear();
	for (int i = 0; i < _vars.size(); i++){
		//update s_val and the deltas
        if(_vars[i]->changed()){
            _s_val.push_back(i);
            updateDelta(i);
        }
		//update s_sup
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
        }
	}
    
	updateTable();
	
	//filterDomains(data,sData->domains,sData->domainSizes);
}


void Table::updateDelta(int i){

    _vars[i]->dump(_vars[i]->min(),_vars[i]->max(),_deltaXs[i]);
    //we calculate the delta by xoring the words
    for (int j = 0; j < _vars[i]->getSizeOfBitSet(); j++){
        _deltaXs[i][j]=_deltaXs[i][j]^_lastVarsValues[i]._words[j].value();
    }
}





void Table::print(){    
    printf("%%%%%% ----------------- TABLE PRINT: -----------------\n\n");

    printf("%%%%%% Size of currTable (no of rows): %ld \n", _tuples.size());
    printf("%%%%%% The table (support) has %ld vars\n",_vars.size());
    for (int i = 0; i < _vars.size(); i++){
        printf("%%%%%% Var size: %d\n",_vars[i]->size());
    }
    int currentOffset=0;
    int internalOffset=0;
    int offsetAccumulator=_vars[0]->intialSize();
	printf("%%%%%% ------ var %d in the table has current size: %d ------\n",currentOffset,_vars[currentOffset]->size());
    
    for (int i = 0; i < _supportSize; i++){
        if(i>=offsetAccumulator){
    		currentOffset++;
    		offsetAccumulator+=_vars[currentOffset]->intialSize();
    		internalOffset=0;
    		printf("%%%%%% ------ var %d in the table has current size: %d ------\n",currentOffset,_vars[currentOffset]->size());
    	}
        _supports[i].print(_variablesOffsets[currentOffset]+internalOffset);
        internalOffset++;
    }
    printf("%%%%%% ----------------- VARS: -----------------\n\n");
    for (int i = 0; i < _vars.size(); i++){
        printf("%%%%%% Var %d: %d\n",i,_vars[i]->getId());

        //_vars[i]->dump(_vars[i]->min(),_vars[i]->max(),_deltaXs[i]);
        
        printf("%%%%%% Var %d dump: %d \n",i,_lastVarsValues[i]._words[0].value());
    }

}