#include <stdio.h>

typedef struct CT{
    
	int variablesNo; //the length of scope 
    char** scope; //variable names
    
    bitSet currTable; //row (bitvector) of current valid rows in the table
    int supportSize; //the length (no of rows) of the supports bitset (CONSTANT)
    bitSet* supports; //table of which values for each variable are required in a constraint
    int* lastSizes; //current domain size of each var
    int* prevLastSizes; //domain size of each var at the previous iteration, from this we could deduce s_val, but still it's used to get delta_x
    //int* s_val; //indexes of the vars not yet instanciated whose domain changed from last iteration (could be replaced by a bitset)
    int* s_sup; //indexes of the vars not yet inst. with at least one value in their domain for which no support has yet been found (could be replaced by a bitset)
    int* residues; //???

	long* supportSizes;//for each var the size of it's domain (CONSTANT), the seizes are the actual sizes (i.e. var 5..7: y; has size 2 not 7)
    
    //for printing purposes only:
    long* variablesOffsets; //offset of the variables, used in accessing the support rows (not all variables start from 0, eg  90..120, variablesOffsets[i]=90 , we avoid the first word, which is still stored for now)
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
