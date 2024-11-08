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


    public:
        SmartTableGPU(vector<var<int>::Ptr> & vars,  vector<std::vector<int>> & tuples, vector<std::vector<int>> & signs);
        void post() override;
        void propagate() override;
};