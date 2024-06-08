#pragma once

#include "global_constraints/table.hpp"
#include <libfca/Types.hpp>
#include <libgpu/Memory.cuh>
#include <libgpu/LinearAllocator.cuh>
#include <libminicpp/varitf.hpp>
#include <libminicpp/constraint.hpp>

class TableGPU : public Table{

    private:
    
    public:
        TableGPU(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples);
        void post() override;
        void propagate() override;
};
