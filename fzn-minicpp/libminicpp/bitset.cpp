/*
 * mini-cp is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * mini-cp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2018. by Laurent Michel, Pierre Schaus, Pascal Van Hentenryck
 * 
 * StaticBitSet Implementation: Tim Curry
 */

#include "bitset.hpp"
#include <string>
#include <stdio.h>
#include <iostream>
using namespace std;
using std::vector;

StaticBitSet::StaticBitSet(int sz)
   : _sz(sz)
{
   int nbWords = (sz >> 5) + ((sz & 0x1f) != 0);
   for (int i = 0; i < nbWords; i++)
      _words.emplace_back(int(0xffffffff));
   if (sz & 0x1f)
      _words[nbWords - 1] = (_words[nbWords - 1] & ~(0xffffffff >> (sz % 32)));
}

bool StaticBitSet::contains(int pos) {
   int word = pos >> 5;  // find word for element to remove
   int shift = pos % 32;  // find pos (from left) within word
   int mask = 0x80000000 >> shift;
   return (_words[word] & mask) != 0;
}

void StaticBitSet::remove(int pos) {
   int word = pos >> 5;  // find word for element to remove
   int shift = pos % 32;  // find pos (from left) within word
   int mask = 0x80000000 >> shift;
   mask = ~mask;
   _words[word] = (_words[word] & mask);
   return;
}

SparseBitSet::SparseBitSet(Trailer::Ptr eng, Storage::Ptr store, int sz)
   : _sz(sz)
{
   _nbWords = (_sz >> 5) + ((_sz & 0x1f) != 0);
   _limit = trail<int>(eng,_nbWords-1);
   for (int i = 0; i < _nbWords; i++) {
      _words.emplace_back(trail<int>(eng, 0xffffffff));
      _index.emplace_back(int(i));
      _mask.emplace_back(int(0));
   }
   if (_sz & 0x1f)
      _words[_nbWords - 1] = (_words[_nbWords - 1] & ~(0xffffffff >> (_sz % 32)));
}

void SparseBitSet::clearMask() {
   int offset;
   for (int i = 0; i <= _limit; i++) {
      offset = _index[i];
      _mask[offset] = 0;
   }
}

void SparseBitSet::reverseMask() {
   int offset;
   for (int i = 0; i <= _limit; i++) {
      offset = _index[i];
      _mask[offset] = ~(_mask[offset]);
   }
}

void SparseBitSet::addToMask(StaticBitSet& m) {
   int offset;
   for (int i = 0; i <= _limit; i++) {
      offset = _index[i];
      _mask[offset] = (_mask[offset] | m[offset]);
   }
}
void SparseBitSet::addToMaskVector(const vector<trail<int>> &v){
   int offset;
   for (int i = 0; i <= _limit; i++) {
      offset = _index[i];
      _mask[offset] = (_mask[offset] | v[i].value());
   }

}
void SparseBitSet::addToMaskInt(unsigned int value){  
	int offset;
   int bitsPerWord=32;
	unsigned int wordToOr=(unsigned int) 1<<(bitsPerWord-(value%bitsPerWord));
	int wordIndex=floor(value/bitsPerWord);
	if(value%bitsPerWord==0){
		wordIndex--;
	}
	offset=_index[wordIndex];
	_mask[offset]=_mask[offset] | wordToOr;
}
void SparseBitSet::intersectWithMask() {
   int offset, w;
   for (int i = _limit; i >= 0; i--) {
      offset = _index[i];
      w = (_words[offset] & _mask[offset]);
      _words[offset] = w;
      if (w == 0) {
         _index[i] = _index[_limit];
         _index[_limit] = offset;
         _limit = _limit - 1;
      }
   }
}

int SparseBitSet::intersectIndex(StaticBitSet& m) {
   int offset;
   for (int i = 0; i <= _limit; i++) {
      offset = _index[i];
      if ((_words[offset] & m[offset]) != 0)
         return offset;
   }
   return -1;
}
int SparseBitSet::intersectIndexSparse(SparseBitSet& m) {
   int offset;
   for (int i = 0; i <= _limit; i++) {
      offset = _index[i];
      if ((_words[offset] & m[offset].value()) != 0)
         return offset;
   }
   return -1;
}

void SparseBitSet::print(int offset) {

    if(_limit.value()>=0)
    	printf("\n%%%%%% Words (for value %d): \n",offset);
      
    else{
    	printf("\n%%%%%% *** Value %d never found in any tuple ***\n",offset);
      return;
   }
    for (int i = 0; i <= _limit.value(); i++) {
        printf("%%%%%% [%d] ", i);
        printBits(_words[i].value());
    }
    if(_limit.value()>=0){
	    printf("%%%%%% Mask: \n");
	    for (int i = 0; i <= _limit.value(); i++) {
	        printf("%%%%%% [%d] ", i);
	        printBits(_mask[i]);
	    }
	}
}
void SparseBitSet::printBits(unsigned int num) {
    // Extracting each bit of the int and printing it
    //yes rather weird function, but since we need to print %%%%%
    vector<char> str = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
    for (int i = 31; i >= 0; i--) {
        str[i] = (num >> i) & 1; 
        printf("%d",str.at(i));
    }
    printf("\n%%%%%% \n");
}

int SparseBitSet::countOnes(){
   
   int count=0;
   for (int i = 0; i <= _nbWords; i++) {
      count+=__builtin_popcount(_words[i].value());
   }
   return count;

}