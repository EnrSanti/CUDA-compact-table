#include "gpu_constriants/table.cuh"

TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){
    printf("%%%%%% TableGPU constructor\n");
}
void TableGPU::post(){

}
void TableGPU::propagate(){

}
