#include "table.hpp"

Table::Table(vector<var<int>::Ptr> & vars, const vector<vector<int>> & tuples) :
    Constraint(vars[0]->getSolver()), 
    _vars(vars), _tuples(tuples), 
    _currTable(SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size())){

    setPriority(CLOW);
    int noTuples=tuples.size();
    int noVars=vars.size();
    _s_val= vector<trail<int>>(noVars);
    _s_sup= vector<trail<int>>(noVars);
    _supportOffsetJmp=vector<int>(noVars);
    
    _variablesOffsets=vector<int>(noVars);
    
    for (int i = 0; i < noVars; i++){
        //intializing the _s_val and _s_sup to 0s
        _s_val[i]=trail<int>(vars[0]->getSolver()->getStateManager(),0);
        _s_sup[i]=trail<int>(vars[0]->getSolver()->getStateManager(),0);
        
        //calculating the number of rows in the support bitset
        _supportSize+=vars[i]->size();
        //we store the offset
        _variablesOffsets[i]=vars[i]->min();

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

            printf("%%%%%% Offset %d\n",offset);
            _supports[offset].addToMaskInt(t+1);
            
            /*_supportsShort[offset].addToMaskInt(t+1);





            for (int varValue = 0; varValue <= entryValue; varValue++) {
                offset=_supportOffsetJmp[v]+varValue;
                //update supportsMin
                _supportsMin[offset].addToMaskInt(t+1);  
            }
            for (int varValue = entryValue; varValue < vars[v]->intialSize(); varValue++) {
                offset=_supportOffsetJmp[v]+varValue;
                //update supportsMax
                _supportsMax[offset].addToMaskInt(t+1);
            }*/
        }
    }
    

    int bitsPerWord=32;

    for (int i = 0; i < _supportSize; ++i){  
        _supports[i].intersectWithMask();
        _supportsShort[i].intersectWithMask();
        _supportsMin[i].intersectWithMask();
        _supportsMax[i].intersectWithMask();

        printf("\n%%%%%% _supports[offset]: %u: \n",_supports[i]._words[0].value());
        _supports[i].clearMask();
        _supportsShort[i].clearMask();
        _supportsMin[i].clearMask();
        _supportsMax[i].clearMask();
        
        //we initialize residues
        for(int j=0; j<noTuples; j++){
            if(_supports[i]._words[j/bitsPerWord]!=0x00000000){
                _residues[i]=trail<int>(_vars[0]->getSolver()->getStateManager(), j); 
            }else{
                _residues[i]=trail<int>(_vars[0]->getSolver()->getStateManager(), 0); 
            }
        }
    }
    print();

}

void Table::post()
{
    for (auto const & v : _vars)
    {
        // v->propagateOnBoundChange(this);
        // v->whenBoundsChange([this, v] {v->removeAbove(0);});
    }
    propagate();
}

void Table::propagate()
{
    printf("%%%%%% Table propagation called.\n");
    // Implement the propagation logic
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

}