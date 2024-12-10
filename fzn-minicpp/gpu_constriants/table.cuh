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
        //int *_supports_mask_dev; //not neeeded, never used 
        unsigned int  * _currTable_dev; //array
        unsigned int  * _currTable_mask_dev; //array
        int * _currTable_size_dev; //just a pointer to a single element
        int * _supportSize_dev; //just a pointer to a single element
        int * _supportOffsetJmp_dev; //array
        int * _variablesOffsets_dev; //array
        int* _svSize_sval_dev;
        int* _svSize_sval_host;
        unsigned int * _vars_dev; //array (matrix) (the domains)
        int * _output_dev; //pointer
        int currTableSize;
        int noVars;
        int *_noVars_dev;
        int *_vars_host;
        unsigned int *_currTable_host;
        int* _outputArray;


        int noBlocks;
        int* offset_dev; //for the ths in the kernel
        int* stream_buffer;
        int noStreams=4; //hardcoded


        int* CTsizes_host;
        int* noBlocks_host;
        int* ss32_host;
        int lastStream_CT;
        int lastStream_SS;
        int lastStream_BL;


        cudaStream_t* streams;

        int *workerOffestAndLimit_dev;
        int *workerOffestAndLimit_host;

    public:
        TableGPU(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples);
        void post() override;
        void propagate() override;
        void enfoceGAC();
        void print();
        void printBits(unsigned int value);
    private:
        int bitsFromRight(int n);
        int bitsFromLeft(int n);
        void dumpDomainsGPU();
        void  divideInStrems(int, int*);
        void varOffsetLimit(int size,int * where);
};

__global__ void printGPUdata(int *_supportSize_dev, int *_variablesOffsets_dev,unsigned int *_currTable_dev,unsigned int *_supports_dev,int * _supportOffsetJmp_dev, int* currTable_size_dev);
__device__ void printBitsGPU(unsigned int num);
__global__ void updateTableGPU(unsigned int* _supports_dev,int * _svSize_off_sval_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* cur_currTable_dev_size, unsigned int* _vars_dev,int* out,int* offsetPerTh, int* offsetsAndLimits);
void printBits(unsigned int num);