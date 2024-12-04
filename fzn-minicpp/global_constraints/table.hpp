#pragma once

#include <libminicpp/varitf.hpp>
#include <libminicpp/bitset.hpp>


using namespace std;





class Table : public Constraint{
    // Constraint private data structures
    protected:
        
        vector<var<int>::Ptr> _vars;
        vector<vector<int>> _tuples;

        SparseBitSet _currTable; 

        int _supportSize; //the length (no of rows) of the supports bitset (CONSTANT)

        vector<SparseBitSet> _supports; //table of which values for each variable are required in a constraint
        
    
        vector<int> _s_val; //indexes of the vars not yet instanciated whose domain changed from last iteration (could be replaced by a bitset)
        vector<int> _s_sup; //indexes of the vars not yet inst. with at least one value in their domain for which no support has yet been found (could be replaced by a bitset)
        vector<trail<int>> _residues; 

        vector<int> _supportOffsetJmp; //for each var the index of the row in "supports" in which such variable starts (CONSTANT)
        
        //c'è in vars[i]->initialMin();
        vector<int> _variablesOffsets; //offset of the variables, used in accessing the support rows (not all variables start from 0, eg  90..120, variablesOffsets[i]=90) 
      
    public:
        Table(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples);
        void post() override;
        void propagate() override;
    protected:
        void enfoceGAC();
        void filterDomains();
        void updateTable();
};


