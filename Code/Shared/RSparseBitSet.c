#define printMaskOff 0
#define printMaskOn 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

//---------------------------------------------
//---------- BITSET DATA DEFINITION -----------
//---------------------------------------------

typedef struct RSBitSet{
	unsigned long* words;
	unsigned long* mask;
	unsigned int* index;
	int limit;
} bitSet;


//---------------------------------------------
//------- CREATE AND DESTROY FUNCTIONS --------
//---------------------------------------------
int bitsPerWord;
//TODO: see variadic macro to use default offset to 0
//function to create the bitset with the specified domain and offset
bitSet createBitSet(int domainSize){

	//calculate the number of words needed to store the domain
	bitsPerWord=(sizeof(unsigned long)*8);
	int wordsNo=ceil(((double)domainSize/bitsPerWord));
	bitSet b;

	//we initialize the fields
	b.words = (unsigned long*)malloc(wordsNo*sizeof(unsigned long));
	b.mask = (unsigned long*)malloc(wordsNo*sizeof(unsigned long));
	b.index = (unsigned int*)malloc(wordsNo*sizeof(unsigned int));
	b.limit = wordsNo;
	for (int i = 0; i < wordsNo-1; i++){
		b.index[i]=i;
		b.mask[i]=0;
		b.words[i]=ULONG_MAX;
	}
	//the last word:
	b.index[wordsNo-1]=wordsNo-1;
	b.mask[wordsNo-1]=0;

	//we calculate the bits to set to 1 in the last word
	b.words[wordsNo-1]=(unsigned long)ULONG_MAX<<(-domainSize%bitsPerWord);

	
	return b;
}

//function to free all the dynamic fields of the bitset passed as input
void freeBitSet(bitSet* b){
	
	free(b->words);
	free(b->mask);
	free(b->index);
}

//---------------------------------------------
//------- BITSET MANIPULATION FUCNTIONS -------
//---------------------------------------------

int isEmpty(bitSet b){
	return (b.limit==-1);
}

void clearMask(bitSet* b){
	//we only reset the parts of the mask useful
	for (int i = 0; i < b->limit; i++){
		b->mask[b->index[i]]=0;
	}
}

void reverseMask(bitSet* b){
	int offset;
	for (int i = 0; i < b->limit; i++){
		offset=b->index[i];
		b->mask[offset]=~(b->mask[offset]);
	}	
}

void addToMask(bitSet* b,unsigned long toAdd[]){
	int offset;
	for (int i = 0; i < b->limit; i++){
		offset=b->index[i];
		b->mask[offset]=b->mask[offset] | toAdd[offset];
	}
}

void intersectWithMask(bitSet* b){
	int offset;
	unsigned long w;
	for (int i = (b->limit)-1; i >= 0; i--){
		offset=b->index[i];
		w=b->words[offset] & b->mask[offset];
		
		if(w!=b->words[offset]){
			b->words[offset]=w;
			if(w==0){
				b->index[i]=b->index[b->limit-1];
				b->index[b->limit-1]=offset;
				(b->limit)--;
			}
		}
	}
}

int intersectIndex(bitSet* b,unsigned long toIntersect[]){
	int offset;
	for (int i = 0; i < b->limit; i++){
		offset=b->index[i];
		if((b->words[offset] & toIntersect[offset])!=0){
			return offset;
		}
	}
	return -1;
}


//---------------------------------------------
//----------- AUX. PRINT FUNCTIONS ------------
//---------------------------------------------
// Function to print the bit representation of a long value
void printLongBits(unsigned long num) {
  
    // Extracting each bit of the double and printing it
    for (int i = 63; i >= 0; i--) {
        long bit = (num >> i) & 1;
        printf("%lu", bit);
    }
    printf("\n");
}
//main print method for the structure

void printBitSet(const bitSet bs,long offset, int includeMask) {
    
    if(bs.limit>0)
    	printf("\nWords (for value %ld): \n",offset);
    else
    	printf("\n*** Value %ld never found in any tuple ***\n",offset);

    for (int i = 0; i < bs.limit; i++) {
        printf("[%d] ", i);
        printLongBits(bs.words[i]);
    }
    if(includeMask==printMaskOn && bs.limit>0){
	    printf("Mask:\n");
	    for (int i = 0; i < bs.limit; i++) {
	        printf("[%d] ", i);
	        printLongBits(bs.mask[i]);

	    }
	}
}


//---------------------------------------------
//-------------- AUX. FUNCTIONS ---------------
//---------------------------------------------

void addToMaskInt(bitSet* b,int value){

	int offset;

	unsigned long wordToOr=(unsigned long) 1<<(bitsPerWord-(value%bitsPerWord));
	int wordIndex=floor(value/bitsPerWord);
	if(value%bitsPerWord==0){
		wordIndex--;
	}
	offset=b->index[wordIndex];
	b->mask[offset]=b->mask[offset] | wordToOr;
	
}
