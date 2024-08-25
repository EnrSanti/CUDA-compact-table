#include "gpu_constriants/smart_table.cuh"

SmartTableGPU::SmartTableGPU(vector<var<int>::Ptr> & vars,  vector<std::vector<int>> & tuples, vector<std::vector<int>> & signs) : SmartTable(vars,tuples,signs){
   printf("%%%%%% hola i am on gpu");

 
}

void SmartTableGPU::post(){
    //printf("%%%%%% post GPU\n");
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void SmartTableGPU::propagate(){
    //printf("%%%%%% propagate on GPU\n");

}
