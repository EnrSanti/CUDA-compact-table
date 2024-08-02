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
    //_supportsMin=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    //_supportsMax=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
   
    //we allocate and initialize the support bitsets
    for (int i = 0; i < _supportSize; i++){
        _supportsShort[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        //_supportsMax[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        //_supportsMin[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
    }

    intializeTable(noVars,noTuples);

}
void SmartTable::intializeTable(int noVars,int noTuples){

    bool found=false;
    int tuplesOfSingletons[noVars];
    for (int v = 0; v < noVars; v++){
        tuplesOfSingletons[v]=-1;
        for (int t = 0; t < noTuples; t++){

            if(_tuples[t][v]>=_vars[v]->min() && _tuples[t][v]<=_vars[v]->max()){
                //classical entry (we update all the supports in the same way)
                

                if(_signs[t][v]==SmartTableOp::All){
                    printf("%%%%%% signs star: %d %d\n",t,v);
                    //set all the bits for the variable in support
                    int offset=_supportOffsetJmp[v];
                    for(int i=0; i<_vars[v]->intialSize(); i++){
                        _supports[offset+i].addToMaskInt(t+1);
                        //don't set anything for supportsShort
                    }
                    //don't set anything for supportsShort
                }else{
                    int entryValue=_tuples[t][v]-_variablesOffsets[v];   
                    int offset=_supportOffsetJmp[v]+entryValue;
                    //if it's not a *, add one bit to both supports
                    _supports[offset].addToMaskInt(t+1); 
                    _supportsShort[offset].addToMaskInt(t+1);
                }
                

                

                /*
                for (int varValue = 0; varValue <= entryValue; varValue++) {
                    offset=_supportOffsetJmp[v]+varValue;
                    //update supportsMin
                    _supportsMin[offset].addToMaskInt(t+1);  
                }
                for (int varValue = entryValue; varValue < _vars[v]->intialSize(); varValue++) {
                    offset=_supportOffsetJmp[v]+varValue;
                    //update supportsMax
                    _supportsMax[offset].addToMaskInt(t+1);
                }*/
                found=true;
                tuplesOfSingletons[v]=t;
            }else{
                _currTable.addToMaskInt(t+1);
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
        //_supportsMin[i].intersectWithMask();
        //_supportsMax[i].intersectWithMask();

        _supports[i].clearMask();
        _supportsShort[i].clearMask();
        //_supportsMin[i].clearMask();
        //_supportsMax[i].clearMask();
        
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
   

    //printing the supports

    for (int i = 0; i < _supportSize; ++i){  
        _supports[i].print(i);
    }
    printf("%%%%%% supportShort:\n");
    for (int i = 0; i < _supportSize; ++i){
        _supportsShort[i].print(i);
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
        // v->propagateOnBoundChange(this);
        // v->whenBoundsChange([this, v] {v->removeAbove(0);});
    }

    propagate();

}

void SmartTable::propagate(){
    printf("%%%%%% Smart Table propagation called.\n");
    // Implement the propagation logic

}