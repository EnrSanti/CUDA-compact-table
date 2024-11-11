
#include "myBitSet.hpp"
#include <string>
#include <stdio.h>
#include <iostream>
using namespace std;
using std::vector;


myBitSet::myBitSet(Trailer::Ptr eng, Storage::Ptr store, int sz) : size(sz) 
{
   int wordsNo = (size >> 5) + ((size & 0x1f) != 0);
   
   limit = trail<int>(eng,wordsNo-1);
   for (int i = 0; i < wordsNo; i++) {
      words.emplace_back(trail<int>(eng, 0xffffffff));
      mask.emplace_back(int(0));
   }
   if (size & 0x1f)
      words[wordsNo - 1] = (words[wordsNo - 1] & ~(0xffffffff >> (size % 32)));
}

void myBitSet::clearMask() {
   int wordsNo=words.size();
   for (int i = 0; i <= wordsNo; i++) {
      mask[i] = 0;
   }
}

void myBitSet::reverseMask() {
   
   int wordsNo=words.size();
   for (int i = 0; i <= wordsNo; i++) {
      mask[i] = ~(mask[i]);
   }
}

void myBitSet::intersectWithMask() {
   int wordsNo=words.size();
   for (int i = wordsNo; i >= 0; i--) {
      words[i].setValue( words[i].value() & mask[i]);
   }
}

int myBitSet::intersectIndex(myBitSet& m) {
   int ret=-1;
   int wordsNo=words.size();
   for (int i = 0; i <= wordsNo; i++) {
      if((words[i].value() & m.words[i].value())!=0){
         ret=i;
      }
   }
   return ret;
}

void myBitSet::addToMaskVector(const vector<trail<int>> &v){
   int wordsNo=words.size();
   for (int i = 0; i <= wordsNo; i++) {
      mask[i] = (mask[i] | v[i].value());
   }

}

void myBitSet::addToMaskInt(unsigned int value){  
	int offset;
    int bitsPerWord=32;
	unsigned int wordToOr=(unsigned int) 1<<(bitsPerWord-(value%bitsPerWord));
   
	int wordIndex=floor(value/bitsPerWord);
	if(value%bitsPerWord==0){
		wordIndex--;
	}
	mask[wordIndex]=mask[wordIndex] | wordToOr;
}

void myBitSet::print() {

   int wordsNo=words.size();
   for (int i = 0; i < wordsNo; i++) {
      printf("%%%%%% [%d] ", i);
      printBits(words[i].value());
   }


}


void myBitSet::printBits(unsigned int num) {
    // Extracting each bit of the int and printing it
    //yes rather weird function, but since we need to print %%%%%
    vector<char> str = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
    for (int i = 31; i >= 0; i--) {
        str[i] = (num >> i) & 1; 
        printf("%d",str.at(i));
    }
    printf("\n%%%%%% \n");
}


int myBitSet::countOnes(){
   
   int count=0;
   int wordsNo=words.size();
   
   for (int i = 0; i < wordsNo; i++) {
      count+=__builtin_popcount(words[i].value());
   }
   return count;

}

bool myBitSet::isEmpty(){
   int wordsNo=words.size();
   for (int i = 0; i < wordsNo; i++) {
      if(words[i].value()!=0){
         return false;
      }
   }
   return true;
}