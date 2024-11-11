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

        unsigned int *supports_dev; //array of arrays linearized
        //int *_supports_mask_dev; //not neeeded, never used 
        unsigned int  * currTable_dev; //array
        unsigned int  * currTable_mask_dev; //array
        int * currTable_size_dev; //just a pointer to a single element
        int * supportSize_dev; //just a pointer to a single element
        int * supportOffsetJmp_dev; //array
        int * variablesOffsets_dev; //array
        int * s_val_size_dev; //pointer
        int * s_val_dev; //array
        unsigned int *vars_dev; //array (matrix) (the domains)
        int *output_dev; //pointer
        int currTableSize;
        int noBlocks;
        int noVars;
        int *noVars_dev;
        int *vars_host;
        int *offset;
        unsigned int *currTable_host;
        int*outputArray;

    public:
        TableGPU(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples);
        void post() override;
        void propagate() override;
        void enfoceGAC();
        void filterDomains();
        void print();
};

__global__ void printGPUdata(int *_supportSize_dev, int *_variablesOffsets_dev,unsigned int *_currTable_dev,unsigned int *_supports_dev,int * _supportOffsetJmp_dev, int* currTable_size_dev);
__device__ void printBitsGPU(unsigned int num);
__global__ void updateTableGPU(unsigned int* _supports_dev,int * _s_val_size_dev, int *_s_val_dev, int *_supportSize_dev, int *_variablesOffsets_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* cur_currTable_dev_size, unsigned int* _vars_dev,int* out, int* offset, int* noVars);
void printBits(unsigned int num);