#include "smart_table.hpp"

enum SmartTableOp {Eq=1, All=2 ,LtInt=3, LtVar=4, GtInt=5, GtVat=6};

SmartTable::SmartTable(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples, vector<vector<int>> & signs) :
    Constraint(vars[0]->getSolver()), _vars(vars), _tuples(tuples), _signs(signs), _currTable(SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size())){

    setPriority(CLOW);
        
    int noTuples=_tuples.size();
    int noVars=_vars.size();

    _s_val= vector<int>();
    _s_sup= vector<int>();
    _supportOffsetJmp=vector<int>(noVars);
    _variablesOffsets=vector<int>(noVars);
    _deltaXs=vector<SparseBitSet>(noVars,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),0));
    _lastVarsValues=vector<SparseBitSet>(noVars,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),0));


    _currTable=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size());

    for (int i = 0; i < noVars; i++){        
        //calculating the number of rows in the support bitset
        _supportSize+=vars[i]->intialSize();
        //we store the offset
        _variablesOffsets[i]=vars[i]->min();
        //we allocate the delta and lastVarsValues
        _deltaXs[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),_vars[i]->size()+1);
        _lastVarsValues[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),_vars[i]->size()+1);

        //initialize lastVarsValues
        vars[i]->dumpInSparseBitSet(vars[i]->min(),vars[i]->max(),_lastVarsValues[i]);
        //printf("%%%%%% intial var values for var %d \n",i);
        //_lastVarsValues[i].printNoMask(0);
    }

    //calculating the offset of the variables, used in accessing the support rows    
    _supportOffsetJmp[0]=0;
    for (int i = 1; i < noVars; i++){
        _supportOffsetJmp[i]=_supportOffsetJmp[i-1]+vars[i-1]->size();
    }

    //we allocate and initialize the support bitsets
    _supports=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _residues= vector<trail<int>>(_supportSize);


    //we allocate and initialize the support bitsets
    for (int i = 0; i < _supportSize; i++){
        _supports[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);//the content doesn't make sense yet, later we need to update the mask and intersect it
    }

    //we allocate and initialize the supports bitsets

    _supportsShort=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _supportsMin=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _supportsMax=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
   
    //we allocate and initialize the support bitsets
    for (int i = 0; i < _supportSize; i++){
        _supportsShort[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        _supportsMax[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        _supportsMin[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
    }

    intializeTable(noVars,noTuples);

}
void SmartTable::intializeTable(int noVars,int noTuples){

    bool found=false;
    int tuplesOfSingletons[noVars];
    for (int v = 0; v < noVars; v++){
        tuplesOfSingletons[v]=-1;
        for (int t = 0; t < noTuples; t++){

            //if we have a * or an entry that is in the domain of the variable, we need to update the supports
            if(_vars[v]->contains(_tuples[t][v])){
                switch(_signs[t][v]) {
                //* entry (we update all the supports in the same way)                
                    case SmartTableOp::All:{
                        printf("%%%%%% signs star: %d %d\n",t,v);

                        //set all the bits for the variable in support
                        int offset=_supportOffsetJmp[v];
                        for(int i=0; i<_vars[v]->intialSize(); i++){
                            _supports[offset+i].addToMaskInt(t+1);
                            _supportsMax[offset+i].addToMaskInt(t+1);
                            _supportsMin[offset+i].addToMaskInt(t+1);
                            //don't set anything for supportsShort
                        }
                        //don't set anything for supportsShort
                        break;
                    }
                    //classical entry
                    default:{
                        int entryValue=_tuples[t][v]-_variablesOffsets[v];   
                        int offset=_supportOffsetJmp[v]+entryValue;
                        //if it's not a *, add one bit to both supports
                        _supports[offset].addToMaskInt(t+1); 
                        _supportsShort[offset].addToMaskInt(t+1);

                        //if the value is not the maximum we need to set the bits in supportsMin
                        for(int i=0; i<_vars[v]->intialSize()-entryValue; i++){
                            _supportsMin[offset+i].addToMaskInt(t+1);
                            //don't set anything for supportsShort
                        }
                        
                        //if the value is not the minimum we need to set the bits in supportsMax
                        for(int i=0; i<=entryValue; i++){
                            _supportsMax[_supportOffsetJmp[v]+i].addToMaskInt(t+1);
                            //don't set anything for supportsShort
                        }
                        
                    }
                }
                
                found=true;
                tuplesOfSingletons[v]=t;
            }else{
                switch(_signs[t][v]) {
                    //* entry (we update all the supports in the same way)                
                    case SmartTableOp::All:{
                        //set all the bits for the variable in support
                        int offset=_supportOffsetJmp[v];

                        for(int i=0; i<_vars[v]->intialSize(); i++){
                            _supports[offset+i].addToMaskInt(t+1);
                            _supportsMax[offset+i].addToMaskInt(t+1);
                            _supportsMin[offset+i].addToMaskInt(t+1);
                            //don't set anything for supportsShort
                        }


                        //don't set anything for supportsShort
                        break;
                    }
                    default:{
                        _currTable.addToMaskInt(t+1);
                    }
                }
            }
        }
        if (found==false){
            //printf("%%%%%% EMPTY DOMAIN\n");
            failNow();
            return;
        }
    }
    
    _currTable.reverseMask();
    _currTable.intersectWithMask();
    _currTable.clearMask();

    
    //printing the supports



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
        bool broken=false;
        for(int j=0; j<noTuples; j++){
            if(_supports[i]._words[j/bitsPerWord].value()!=0x00000000 && !broken){
                _residues[i]=trail<int>(_vars[0]->getSolver()->getStateManager(), j); 
                broken=true;
            }else{
                _residues[i]=trail<int>(_vars[0]->getSolver()->getStateManager(), 0); 
            }
        }
    }
   

    //printing the supports //to remove
    printf("%%%%%% supportSize: %d\n",_supportSize);
    for (int i = 0; i < _supportSize; ++i){  
        _supports[i].print(i);
    }
    printf("%%%%%% supportShort:\n");
    for (int i = 0; i < _supportSize; ++i){
        _supportsShort[i].print(i);
    }
    
    printf("%%%%%% supportMin:\n");
    for (int i = 0; i < _supportSize; ++i){
        _supportsMin[i].print(i);
    }
    printf("%%%%%% supportMax:\n");
    for (int i = 0; i < _supportSize; ++i){
        _supportsMax[i].print(i);
    }
   
   
    //forall vars
    for (int i = 0; i < noVars; i++){
        if(_vars[i]->size()==1){
            if(tuplesOfSingletons[i]==-1){
                //printf("%%%%%% EMPTY DOMAIN 2\n");
                failNow();
                return;
            }
            _currTable.addToMaskInt(tuplesOfSingletons[i]+1);
            _currTable.intersectWithMask();
            _currTable.clearMask();
        }
    }
    if(_currTable.isEmpty()){
        //printf("%%%%%% EMPTY DOMAIN 3\n");
        failNow();
        return;
    }
}
void SmartTable::post(){
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}

void SmartTable::propagate(){
    enfoceGAC();
}



//---------------------------------------------
//------- The three functions of alg. 2 -------
//---------------------------------------------

void SmartTable::updateTable(){
    //forall var x in s_val
    int index=0;
    
    for(int i=0; i < _s_val.size(); ++i){
        _currTable.clearMask();
        index=_s_val[i];
        if(_deltaXs[index].countOnes() < _vars[index]->size()){//_deltaXs[index].countOnes() < _vars[index]->size()
            //incremental update
             //print the words of the delta
          
            for (int j = 0; j < _vars[index]->intialSize(); j++){
                //printf("%%%%%% deltaXs[%d] contains 1 at pos %d? ",index,_vars[index]->initialMin()+j);  
                if(_deltaXs[index].getIthBit(j+_vars[index]->initialMin())==1){     
                    int index_x_a=_supportOffsetJmp[index]+j;
                    _currTable.addToMaskVector(_supportsShort[index_x_a]._words);
                }
            }    

            _currTable.reverseMask();

            // TODO UNCOMMENT
            //if(dom(i).minChanged()){
            //	...	
            //}
            //if(dom(i).maxChanged()){
            //	...
            //}
        }else{
            //reset based update
            //printf("%%%%%% reset based update \n");
            vector<int> dom=_vars[index]->dumpDomainToVec();
            
            for (int j = 0; j < dom.size(); j++){ 
                int index_x_a=_supportOffsetJmp[index]+dom[j]-_variablesOffsets[index];
                _currTable.addToMaskVector(_supports[index_x_a]._words);
            } 
        }

        _currTable.intersectWithMask();

        if(_currTable.isEmpty()){
            //printf("%%%%%% Table is empty, backtrack\n");
            failNow();
            return;
		}
    }

}

void SmartTable::filterDomains(){
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
        //printf("%%%%%% new domain for var %d\n",index);
        _vars[index]->dumpInSparseBitSet(_vars[index]->min(),_vars[index]->max(),_lastVarsValues[index]);
        //_lastVarsValues[index].printNoMask(0);
    }
}

void SmartTable::enfoceGAC(){
    //update the table
    
    _s_val.clear();
    _s_sup.clear();
	for (int i = 0; i < _vars.size(); i++){
		//update s_val and the deltas
        if(_vars[i]->changed()){
            _s_val.push_back(i);
            //printf("%%%%%% Var %d changed UPDATING DELTA\n",i);
            updateDelta(i);
        }
		//update s_sup
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
        }
	}
	updateTable();
	
	filterDomains();
    
}


void SmartTable::updateDelta(int i){

    _vars[i]->dumpInSparseBitSet(_vars[i]->min(),_vars[i]->max(),_deltaXs[i]);
    /*
    printf("%%%%%% curr domain for var %d \n",i);
    _deltaXs[i].printNoMask(0);
    printf("%%%%%% last var values for var %d \n",i);
    _lastVarsValues[i].printNoMask(0);
    */
    //we calculate the delta by xoring the words
    for (int j = 0; j < _vars[i]->getSizeOfBitSet(); j++){
        _deltaXs[i]._words[j].setValue(_deltaXs[i]._words[j].value()^_lastVarsValues[i]._words[j].value()); //BEWARE, BROKEN THE DATA STRUCTURE can be replaced with x XOR y = (x AND (NOT y)) OR ((NOT x) AND y)
    }
    //printf("%%%%%% this DELTA contain the changes %d \n",i);
    //_deltaXs[i].printNoMask(0);
}
