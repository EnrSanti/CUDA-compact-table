#pragma once

#include <libminicpp/varitf.hpp>
#include <libminicpp/bitset.hpp>
using namespace std;


class SmartTable : public Constraint {

    // Constraint private data structures
    protected:

        vector<var<int>::Ptr> _vars;
        vector<std::vector<int>> _tuples;
        vector<std::vector<int>> _signs;
        

        SparseBitSet _currTable; 

        int _supportSize; //the length (no of rows) of the supports bitset (CONSTANT)

        unsigned int *_supports; //table of which values for each variable are required in a constraint
        int currTableSize;
    

        vector<int> _s_val; //indexes of the vars not yet instanciated whose domain changed from last iteration (could be replaced by a bitset)
        vector<int> _s_sup; //indexes of the vars not yet inst. with at least one value in their domain for which no support has yet been found (could be replaced by a bitset)
       
        vector<int> _supportOffsetJmp; //for each var the index of the row in "supports" in which such variable starts (CONSTANT)
       
        vector<int> _variablesOffsets;

    public:
        SmartTable(vector<var<int>::Ptr> & vars,  vector<std::vector<int>> & tuples, vector<std::vector<int>> & signs);
        void post() override;
        void propagate() override;
    protected:
        void intializeTable(int, int);
        void enfoceGAC();
        void updateTable();
        void filterDomains();
        void addToMaskInt(unsigned int* mask,int value);
        int intersectIndexSparse(unsigned int* words,SparseBitSet& m);
        
};
