#pragma once

#include <libminicpp/varitf.hpp>
#include <libminicpp/bitset.hpp>


using namespace std;



/*
int variablesNo; //the length of scope  -> we have them in vars
char** scope; //variable names -> we have them in vars

bitSet currTable; //row (bitvector) of current valid rows in the table -> StaticBitSet
int supportSize; //the length (no of rows) of the supports bitset (CONSTANT)


bitSet* supports; //table of which values for each variable are required in a constraint
bitSet* supportsShort; //additional bitset to deal with short tables, bitset value to 1 iff (x,a) strictly accepted by the i-th tuple     //(in the paper they are supports*)
bitSet* supportsMin; //additional bitset to deal with <= and < (smart tables)
bitSet* supportsMax; //additional bitset to deal with >= and > (smart tables)



int* lastSizes; //current domain size of each var
int* s_val; //indexes of the vars not yet instanciated whose domain changed from last iteration (could be replaced by a bitset)
int* s_sup; //indexes of the vars not yet inst. with at least one value in their domain for which no support has yet been found (could be replaced by a bitset)
long* residues; 


long* supportSizes; //for each var the size of it's domain (CONSTANT), the sizes are the actual sizes (i.e. var 5..7: y; has size 3 not 7 as if was starting from 0)
long* supportOffsetJmp; //for each var the index of the row in "supports" in which such variable starts (CONSTANT)


long* variablesOffsets; //offset of the variables, used in accessing the support rows (not all variables start from 0, eg  90..120, variablesOffsets[i]=90) 

*/



class Table : public Constraint{
    // Constraint private data structures
    public:
        
        vector<var<int>::Ptr> _vars;
        vector<vector<int>> _tuples;
        SparseBitSet _currTable; 
    
        int _supportSize; //the length (no of rows) of the supports bitset (CONSTANT)

        vector<SparseBitSet> _supports; //table of which values for each variable are required in a constraint
        vector<SparseBitSet> _supportsShort; //additional bitset to deal with short tables, bitset value to 1 iff (x,a) strictly accepted by the i-th tuple     //(in the paper they are supports*)
        vector<SparseBitSet> _supportsMin; //additional bitset to deal with <= and < (smart tables)
        vector<SparseBitSet> _supportsMax; //additional bitset to deal with >= and > (smart tables)
        
        //già l'abbiamo in var[i]->size()
        //int* lastSizes; //current domain size of each var 

        vector<trail<bool>> _s_val; //indexes of the vars not yet instanciated whose domain changed from last iteration (could be replaced by a bitset)
        vector<trail<bool>> _s_sup; //indexes of the vars not yet inst. with at least one value in their domain for which no support has yet been found (could be replaced by a bitset)
        vector<trail<int>> _residues; 

        //già l'abbiamo in var[i]->InitialSize()
        //vector<long> supportSizes; //for each var the size of it's domain (CONSTANT), the sizes are the actual sizes (i.e. var 5..7: y; has size 3 not 7 as if was starting from 0)
        
        
        vector<int> _supportOffsetJmp; //for each var the index of the row in "supports" in which such variable starts (CONSTANT)
        vector<int> _variablesOffsets; //offset of the variables, used in accessing the support rows (not all variables start from 0, eg  90..120, variablesOffsets[i]=90) 
        
    public:
        Table(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples);
        void post() override;
        void propagate() override;
        void print();
    private:
        void enfoceGAC();
};


