#include "gpu_constriants/smart_table.cuh"

SmartTableGPU::SmartTableGPU(vector<var<int>::Ptr> & vars,  vector<std::vector<int>> & tuples, vector<std::vector<int>> & signs) : SmartTable(vars,tuples,signs){
   setPriority(CLOW);
 
}
void SmartTableGPU::post(){}
void SmartTableGPU::propagate(){};