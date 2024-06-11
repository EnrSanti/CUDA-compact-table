#pragma once

#include "global_constraints/table.hpp"
#include <libfca/Types.hpp>
#include <libgpu/Memory.cuh>
#include <libgpu/LinearAllocator.cuh>
#include <libminicpp/varitf.hpp>
#include <libminicpp/constraint.hpp>
using namespace std;
using namespace Fca;
using namespace Gpu::Memory;

class TableGPU : public Table{

    private:
        u32 sm_count;

        int **_supports_dev; //array of arrays
        //int *_supports_mask_dev; //not neeeded, never used array
        int * _currTable_dev; //array
        int * _currTable_mask_dev; //array
        int * _supportSize_dev; //just a pointer to a single element
        int * _variablesOffsets_dev; //array
    public:
        TableGPU(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples);
        void post() override;
        void propagate() override;
};

__global__ void printGPUdata(int *_supportSize_dev, int* _variablesOffsets_dev,SparseBitSet *_currTable_dev,SparseBitSet *_supports_dev);
__device__ void printNoMask(int offset,SparseBitSet *_bitSet_dev);
__device__ void printBits(unsigned int num);