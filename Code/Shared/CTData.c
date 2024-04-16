#include <stdio.h>

typedef struct CT{
    
	int variablesNo; //the length of scope 
    char** scope; //variable names
    
    bitSet currTable; //row (bitvector) of current valid rows in the table
    int supportSize; //the length (no of rows) of the supports bitset (CONSTANT)
    bitSet* supports; //table of which values for each variable are required in a constraint
    int* lastSizes; //current domain size of each var
    int* s_val; //indexes of the vars not yet instanciated whose domain changed from last iteration (could be replaced by a bitset)
    int* s_sup; //indexes of the vars not yet inst. with at least one value in their domain for which no support has yet been found (could be replaced by a bitset)
    long* residues; 

	long* supportSizes;//for each var the size of it's domain (CONSTANT), the seizes are the actual sizes (i.e. var 5..7: y; has size 3 not 7)
    long* supportOffsetJmp; //for each var the index of the row in supports in which such variable starts(CONSTANT)

    long* variablesOffsets; //offset of the variables, used in accessing the support rows (not all variables start from 0, eg  90..120, variablesOffsets[i]=90) todo: for now just used for printing purposes, the first word is still saved

} CT;

void printCT(const CT *ct) {
     
  
    printf("Variables:");

    for (int i = 0; i < ct->variablesNo; i++) {
        printf("%s ", ct->scope[i]);
    }


    printf("\ncurrent table:\n");
	printBitSet(ct->currTable,0,printMaskOff);

    printf("\n----------------------------\n");

    printf("supports: \n");
    long currentOffset=0;
    long internalOffset=0;
    long offsetAccumulator=ct->supportSizes[0];
	printf("--- var: %s, size: %ld ---\n",ct->scope[currentOffset],ct->supportSizes[currentOffset]);
    for (int i = 0; i < ct->supportSize; ++i){
    	if(i>=offsetAccumulator){
    		currentOffset++;
    		offsetAccumulator+=ct->supportSizes[currentOffset];
    		internalOffset=0;
    		printf("--- var: %s, size: %ld ---\n",ct->scope[currentOffset],ct->supportSizes[currentOffset]);
    	}
		printBitSet(ct->supports[i],ct->variablesOffsets[currentOffset]+internalOffset,printMaskOff);	
    	
    	internalOffset++;
    }
   
}
