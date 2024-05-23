#include "table.hpp"

Table::Table(vector<var<int>::Ptr> & vars, const vector<vector<int>> & tuples) :
    Constraint(vars[0]->getSolver()), 
    _vars(vars), _tuples(tuples), 
    _currTable(SparseBitSet(vars[0]->getSolver()->getStateManager(),nullptr,tuples.size())){

    setPriority(CLOW);
    int noTuples=tuples.size();
    printf("%%%%%% Table constructor called\n");
    printf("%%%%%% size of _currTable: %d \n", noTuples);
    for (int i = 0; i < vars.size(); i++){
        printf("%%%%%% var size: %d\n",vars[i]->size());
        _supportSize+=vars[i]->size();
    }

    //initialize the supports vector
    _supports=vector<SparseBitSet>(vars.size(),SparseBitSet(vars[0]->getSolver()->getStateManager(),nullptr,_supportSize));
    _supportsShort=vector<SparseBitSet>(vars.size(),SparseBitSet(vars[0]->getSolver()->getStateManager(),nullptr,_supportSize));
    _supportsMin=vector<SparseBitSet>(vars.size(),SparseBitSet(vars[0]->getSolver()->getStateManager(),nullptr,_supportSize));
    _supportsMax=vector<SparseBitSet>(vars.size(),SparseBitSet(vars[0]->getSolver()->getStateManager(),nullptr,_supportSize));
    

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
