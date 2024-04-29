#include <stdio.h>

typedef struct CT{
    
	int variablesNo; //the length of scope 
    char** scope; //variable names
    
    bitSet currTable; //row (bitvector) of current valid rows in the table
    int supportSize; //the length (no of rows) of the supports bitset (CONSTANT)
    bitSet* supports; //table of which values for each variable are required in a constraint
    bitSet* supportsShort; //additional bitset to deal with short tables, bitset value to 1 iff (x,a) strictly accepted by the i-th tuple     //(in the paper they are supports*)
    int* lastSizes; //current domain size of each var
    int* s_val; //indexes of the vars not yet instanciated whose domain changed from last iteration (could be replaced by a bitset)
    int* s_sup; //indexes of the vars not yet inst. with at least one value in their domain for which no support has yet been found (could be replaced by a bitset)
    long* residues; 

	long* supportSizes; //for each var the size of it's domain (CONSTANT), the sizes are the actual sizes (i.e. var 5..7: y; has size 3 not 7 as if was starting from 0)
    long* supportOffsetJmp; //for each var the index of the row in "supports" in which such variable starts (CONSTANT)

    long* variablesOffsets; //offset of the variables, used in accessing the support rows (not all variables start from 0, eg  90..120, variablesOffsets[i]=90) 

} CT;
void printCurrTable(const CT *ct){
    printf("\ncurrent table:");
    printBitSet(ct->currTable,0,printMaskOff,ANSI_COLOR_RESET);
}
void printCTData(const CT *ct) {

  
    printf("Variables:");

    for (int i = 0; i < ct->variablesNo; i++) {
        printf("%s ", ct->scope[i]);
    }


    printCurrTable(ct);
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
    		printf("\n--- var: %s, size: %ld ---\n",ct->scope[currentOffset],ct->supportSizes[currentOffset]);
    	}
		printBitSet(ct->supports[i],ct->variablesOffsets[currentOffset]+internalOffset,printMaskOff,ANSI_COLOR_GREEN);	
    	printBitSet(ct->supportsShort[i],ct->variablesOffsets[currentOffset]+internalOffset,printMaskOff,ANSI_COLOR_GREEN);  
        
    	internalOffset++;
    }
   
}