#include <stdio.h>

typedef struct CT{
	int variablesNo; //the length of scope 
    long* variablesOffsets; //offset of the variables, used in accessing the support rows (not all variables start from 0, eg  90..120, we avoid the first word)
    char** scope; //variable names
    bitSet currTable; //row (bitvector) of current valid rows in the table
    int supportSize; //the length (rows) of the supports
    bitSet* supports; //table of which values for each variable are required in a constraint
    int* lastSizes; 
    char** s_val;
    char** s_sup;
    int* residues;

    //for printing purposes only:
	long* supportSizes;
} CT;


void printCT(const CT *ct) {
    printf("Variables:");

    for (int i = 0; i < ct->variablesNo; i++) {
        printf("%s ", ct->scope[i]);
    }


    printf("\ncurrent table:\n");
	printBitSet(ct->currTable,0,printMaskOff);

    printf("\n----------------------------\n");

    printf("supports:\n");
    long currentOffset=0;
    long internalOffset=0;
    long offsetAccumulator=ct->supportSizes[0];
	printf("--- var: %s ---\n",ct->scope[currentOffset]);
    for (int i = 0; i < ct->supportSize; ++i){
    	if(i>=offsetAccumulator){
    		currentOffset++;
    		offsetAccumulator+=ct->supportSizes[currentOffset];
    		internalOffset=0;
    		printf("--- var: %s ---\n",ct->scope[currentOffset]);
    	}
		printBitSet(ct->supports[i],ct->variablesOffsets[currentOffset]+internalOffset,printMaskOff);	
    	
    	internalOffset++;
    }
}
