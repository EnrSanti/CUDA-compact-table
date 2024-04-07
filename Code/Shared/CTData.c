#include <stdio.h>

typedef struct CT{
	int variablesNo; //the length of scope 
    char** scope; //variable names
    bitSet currTable; //row (bitvector) of current valid rows in the table
    int supportSize; //the length (rows) of the supports
    bitSet* supports; //table of which values for each variable are required in a constraint
    int* lastSizes; //
    char** s_val;
    char** s_sup;
    int* residues;
} CT;


void printCT(const CT *ct) {
    printf("Variables:");

    for (int i = 0; i < ct->variablesNo; i++) {
        printf("%s, ", ct->scope[i]);
    }


    printf("\ncurrent table:\n");
	printBitSet(ct->currTable);

    printf("\n----------------------------\n");

    printf("supports:\n");
    for (int i = 0; i < ct->supportSize; ++i){
		printBitSet(ct->supports[i]);	
    }
}
