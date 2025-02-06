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

        unsigned int *_supports_dev; //array of arrays linearized
        int * _currTable_size_dev; //just a pointer to a single element
        int * _supportSize_dev; //just a pointer to a single element
        int * _supportOffsetJmp_dev; //array
        int * _variablesOffsets_dev; //array
        int * _vars_dev; //array (matrix) (the domains)
        int currTableSize;
        int noVars;
        int *_noVars_dev;
        int *_vars_host;
        unsigned int * _vars_to_remove_host;
        unsigned int * _tmpMasks;

        unsigned int* _CT_MASKCT_svSize_sval_sSize_sSup_host;
        unsigned int* _CT_MASKCT_svSize_sval_sSize_sSup_dev;
        

        int noBlocks;
        int noBlocksFilter;
        const int noStreams=1; //hardcoded, don't touch IN THIS BRANCH

        cudaStream_t* streams;

        int *workerOffestAndLimit_dev;
        int internalIndex;
        int *workerOffestAndLimit_host;

        unsigned int* buffer; //just for the new dump
        int buffSize;

    public:
        TableGPU(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples);
        void post() override;
        void propagate() override;
        void print();
    private:
        void dumpDomainsGPU2();
};

__global__ void updateTableGPU(unsigned int* _supports_dev,unsigned int * _svSize_off_sval_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* _currTable_dev_size, int* _vars_dev, int* offsetsAndLimits, unsigned int* _tmpMasks);
__global__ void reduce(unsigned int* _CT_MASKCT_svSize_sval_sSize_sSup_dev,unsigned int* _tmpMasks,int* _currTable_size_dev);
    
void varOffsetLimit(int size,int * where);
int bitsFromRight(int n);
int bitsFromLeft(int n);
