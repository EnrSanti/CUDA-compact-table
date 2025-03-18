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
        
        unsigned int* _CT_mask_svs_host;
        unsigned int* _CT_mask_svs_dev;

        int noBlocks;
        int noBlocksFilter;
        const int noStreams=1; //hardcoded, don't touch IN THIS BRANCH

        cudaStream_t* streams;

        int *doms_doms_before_dev;
        int internalIndex;
        int *doms_doms_before_host;

        unsigned int* buffer; //just for the new dump
        
        unsigned int* _supportsT_host;
        unsigned int* _supportsT_dev;
        int* noTuples_dev; 

    public:
        SmartTableGPU(vector<var<int>::Ptr> & vars,  vector<std::vector<int>> & tuples, vector<std::vector<int>> & signs);
        void post() override;
        void propagate() override;
        void print();
    private:
        void dumpDomainsGPU2();
};