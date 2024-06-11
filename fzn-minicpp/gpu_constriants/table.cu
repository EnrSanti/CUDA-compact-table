#include "gpu_constriants/table.cuh"
using namespace std;
using namespace Fca;
using namespace Gpu::Memory;

TableGPU::TableGPU(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) : Table(vars,tuples){
    setPriority(CLOW);
    printf("%%%%%% TableGPU constructor\n");

    int noTuples=tuples.size();
    int noVars=vars.size();


    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    sm_count = device_prop.multiProcessorCount;
    int cores_per_SM = 128;
    printf("%%%%%% number of SMs: %d\n",sm_count);
    printf("%%%%%% warp size: %d\n",32);
    printf("%%%%%% cores per SM: %d\n",cores_per_SM);
    printf("%%%%%% support size: %d\n",_supportSize);
    
    
    // Memory allocation
    //_supports_dev = mallocDevice<SparseBitSet>(sizeof(SparseBitSet)*_supportSize);
    _currTable_dev = mallocDevice<int>(sizeof(int)); //vedi per cosa moltiplicare
    //printing currTable info
    printf("%%%%%% currTable: %u\n", _currTable._words.size());
    _currTable.print(0);
    _supportSize_dev = mallocDevice<int>(sizeof(int));
    _variablesOffsets_dev = mallocDevice<int>(sizeof(int)*noVars);
    printf("%%%%%% Memory allocation on device\n");


    //Memory copy
    //cudaMemcpyAsync(_supports_dev, _supports.data(), sizeof(SparseBitSet)*_supportSize, cudaMemcpyHostToDevice);
    //(_currTable_dev, &_currTable, sizeof(SparseBitSet), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_supportSize_dev, &_supportSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(_variablesOffsets_dev, _variablesOffsets.data(), sizeof(int)*noVars, cudaMemcpyHostToDevice);
    printf("%%%%%% After mem cpy\n");
    //printGPUdata<<<1,1>>>(_supportSize_dev,_variablesOffsets_dev,_currTable_dev,_supports_dev);
    cudaDeviceSynchronize();
}
void TableGPU::post(){
    printf("%%%%%% post GPU\n");
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}
void TableGPU::propagate(){
    printf("%%%%%% propagate on GPU\n");
}
__global__ void printGPUdata(int *_supportSize_dev, int *_variablesOffsets_dev,SparseBitSet *_currTable_dev,SparseBitSet *_supports_dev){
    printf("%%%%%% -------------------------- printGPUdata -------------------------- \n");
    printf("%%%%%% threadIdx.x: %d\n",threadIdx.x);
    printf("%%%%%% _supportSize_dev: %d\n",*_supportSize_dev);
    
    for(int i=0;i<*_supportSize_dev;i++){
        printNoMask(i,_supports_dev); //NON è i ma ok
    }
 
}
__device__ void printNoMask(int offset, SparseBitSet *_bitSet_dev) {
  
}
__device__ void printBits(unsigned int num) {
    // Extracting each bit of the int and printing it
    //yes rather weird function, but since we need to print %%%%%
    char str[32] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
    for (int i = 31; i >= 0; i--) {
        str[i] = (num >> i) & 1; 
        printf("%d",str[i]);
    }
    printf("\n%%%%%% \n");
}