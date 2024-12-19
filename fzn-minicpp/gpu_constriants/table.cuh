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

        unsigned int *_supports_dev; //array of arrays linearized
        int * _currTable_size_dev; //just a pointer to a single element
        int * _supportSize_dev; //just a pointer to a single element
        int * _supportOffsetJmp_dev; //array
        int * _variablesOffsets_dev; //array
        unsigned int * _vars_dev; //array (matrix) (the domains)
        int currTableSize;
        int noVars;
        int *_noVars_dev;
        int *_vars_host;




        unsigned int* _CT_MASKCT_svSize_sval_sSize_sSup_host;
        unsigned int* _CT_MASKCT_svSize_sval_sSize_sSup_dev;


        int noBlocks;
        int noBlocksEmpty;
        const int noStreams=1; //hardcoded, don't touch IN THIS BRANCH


        cudaStream_t* streams;

        int *workerOffestAndLimit_dev;
        int internalIndex;
        int *workerOffestAndLimit_host;


    public:
        TableGPU(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples);
        void post() override;
        void propagate() override;
        void enfoceGAC();
        void print();
    private:
        int bitsFromRight(int n);
        int bitsFromLeft(int n);
        void dumpDomainsGPU();
        void varOffsetLimit(int size,int * where);
        void enfGACDev();
};

__global__ void printGPUdata(int *_supportSize_dev, int *_variablesOffsets_dev,unsigned int *_currTable_dev,unsigned int *_supports_dev,int * _supportOffsetJmp_dev, int* currTable_size_dev, unsigned int* domains);
__device__ void printBitsGPU(unsigned int num);
__global__ void updateTableGPU(unsigned int* _supports_dev,unsigned int * _svSize_off_sval_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* _currTable_dev_size, unsigned int* _vars_dev, int* offsetsAndLimits);
__global__ void  filterDomainsGPU();
__global__ void  intersectGPU(unsigned int* _CT_MASKCT_svSize_sval_sSize_sSup_dev, int* _currTable_dev_size);