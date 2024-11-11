#pragma once

#include "global_constraints/smart_table.hpp"
#include <libfca/Types.hpp>
#include <libgpu/Memory.cuh>
#include <libgpu/LinearAllocator.cuh>
#include <libminicpp/varitf.hpp>
#include <libminicpp/constraint.hpp>

using namespace std;
using namespace Fca;
using namespace Gpu::Memory;

class SmartTableGPU : public SmartTable {

    // Constraint private data structures
    private:
        u32 sm_count;

        unsigned int *_supports_dev; //array of arrays linearized
        //************************************
        //if no incremental update is used the following three arrays are USELESS
        //************************************
        unsigned int *_supportsShort_dev;
        unsigned int *_supportsMin_dev;
        unsigned int *_supportsMax_dev;

        //int *_supports_mask_dev; //not neeeded, never used 
        unsigned int  * _currTable_dev; //array
        unsigned int  * _currTable_mask_dev; //array
        int * _currTable_size_dev; //just a pointer to a single element
        int * _supportSize_dev; //just a pointer to a single element
        int * _supportOffsetJmp_dev; //array
        int * _variablesOffsets_dev; //array
        int * _s_val_size_dev; //pointer
        int * _s_val_dev; //array
        unsigned int * _vars_dev; //array (matrix) (the domains)
        int * _output_dev; //pointer
        int currTableSize;
        int noVars;
        int *_noVars_dev;
        int *_vars_host;
        int *_offset;
        unsigned int *_currTable_host;
        int* _outputArray;

    public:
        SmartTableGPU(vector<var<int>::Ptr> & vars,  vector<std::vector<int>> & tuples, vector<std::vector<int>> & signs);
        void post() override;
        void propagate() override;
        void enfoceGAC() ;
        void filterDomains();
};
__global__ void updateSmartTableGPU(unsigned int* _supports_dev,int * _s_val_size_dev, int *_s_val_dev, int *_supportSize_dev, int *_variablesOffsets_dev, int *_supportOffsetJmp_dev, unsigned int * _currTable_dev,int* cur_currTable_dev_size, unsigned int* _vars_dev,int* out, int* offset, int* noVars);