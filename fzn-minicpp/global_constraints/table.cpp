#include "table.hpp"

Table::Table(vector<var<int>::Ptr> & vars, const vector<vector<int>> & tuples) :
    Constraint(vars[0]->getSolver()), 
    _vars(vars), _tuples(tuples), 
    _currTable(SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size())){

    setPriority(CLOW);
    int noTuples=tuples.size();
    printf("%%%%%% Table constructor called\n");
    printf("%%%%%% size of _currTable: %d \n", noTuples);
    int noVars=vars.size();
    printf("%%%%%% the table has %d vars\n",noVars);

    _s_val= vector<trail<int>>(noVars);
    _s_sup= vector<trail<int>>(noVars);
    _supportOffsetJmp=vector<int>(noVars);
    
    _variablesOffsets=vector<int>(noVars);
    
    for (int i = 0; i < noVars; i++){
        printf("%%%%%% var size: %d\n",vars[i]->size());
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
        
    }
    */

    //we allocate and initialize the support bitsets
    for (int i = 0; i < _supportSize; i++){
        _supports[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);//the content doesn't make sense yet, later we need to update the mask and intersect it
        _supportsShort[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        _supportsMax[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        _supportsMin[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
    }





















    // Examples:
    // Initialization backtrackable int vector: [3,3,3,3,3,3,3,3,3,3]
    //for (int i = 0; i < 10; i  += 1)
    //{
    //    biv.push_back(trail<int>(x[0]->getSolver()->getStateManager(), 3));
    //}
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
