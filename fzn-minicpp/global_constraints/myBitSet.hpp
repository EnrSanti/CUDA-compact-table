
#ifndef __MY_BITSET_H
#define __MY_BITSET_H

#include "../libminicpp/trailable.hpp"
#include "../libminicpp/handle.hpp"
#include "../libminicpp/store.hpp"
#include <vector>
#include <cmath>

class myBitSet {
private:
   void printBits(unsigned int num);
   int size;
   trail<int> limit;
  

public:
   std::vector<int>            mask;  
   std::vector<trail<int>>     words; 
   myBitSet(Trailer::Ptr eng, Storage::Ptr store, int size);
   bool isEmpty();
   void clearMask();
   void reverseMask();
   void addToMaskVector(const std::vector<trail<int>> &v);
   void addToMaskInt(unsigned int value);
   void intersectWithMask();
   int intersectIndex(myBitSet& m);
   trail<int>& operator[] (int i) { return words[i];}
   int operator[] (int i) const { return words[i].value();}
   void print();
   int countOnes(); //"size"
};

#endif
