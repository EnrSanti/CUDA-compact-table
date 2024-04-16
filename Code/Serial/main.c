#include <stdio.h>
#include "constraintCT.c"
#include <math.h>


//data to random remove supports
int seed=1273;
int maxValuesToRemove=3;
solverData sData;

int hasSolution(CT table){
	return !isEmpty((table.currTable));
}

int* genNoSupportsVals(CT *table, int maxValuesToRemove){
	int lowerbound= (maxValuesToRemove < table->variablesNo) ? maxValuesToRemove : table->variablesNo;
	for (int i = 0; i < table->variablesNo; i++) {
        sData.deltaXSizes[i] = rand() % (lowerbound + 1); //random no between 0 and maxValuesToRemove
    }
}

void printValsToRemove(CT *table){
	printf("\n-----------------------------------\n");
	for (int i = 0; i < table->variablesNo; i++){
		printf("Remove from %s: ",table->scope[i]);
		for (int j = 0; j < sData.deltaXSizes[i]; j++){
			printf("%d, ",sData.deltaXs[i][j]);
		}
		printf("\n");
	}
	printf("-----------------------------------\n");
	
}

void RemoveRandomSupports(CT *table){

	long minValueInSupport;
	long maxValueInSupport;
	int toRemove;
	//for each var we seek some value to remove
	genNoSupportsVals(table,maxValuesToRemove); 
	for (int i = 0; i < table->variablesNo; i++){
		minValueInSupport=table->variablesOffsets[i];
		maxValueInSupport=minValueInSupport+table->supportSizes[i]-1;
		for (int j = 0; j < sData.deltaXSizes[i]; j++){
			//we now generate the actual values to remove
			sData.deltaXs[i][j]=(rand() % (maxValueInSupport - minValueInSupport + 1)) + minValueInSupport; //random no between min and max
			sData.domains[i][j]=0;
			//todo 1 volta funziona poi no (ovviamente)
		}
		sData.domainSizes[i]-=sData.deltaXSizes[i];

		//todo doens't work this way
	}
	printValsToRemove(table);
}

void randomSolve(CT *table,int iterations){
	for (int i = 0; i < iterations; i++)	{
		if(!hasSolution(*table)){
			printf("\n--------------- The problem has no solution ---------------\n");
			return;
		}
		RemoveRandomSupports(table);
		enfoceGAC(table,&sData); //we discard for now the return value (backtrack, todo: use it)
		printCurrTable(table);
	}
}



void main(int argc, char const* argv[]) {

	//we initialize the random generator
	srand(seed);

    //if the user didn't insert the file path or typed more
    if (argc != 2) {
        printf("Insert the file path\n");
        return;
    }

    //create the strucure and we instanciate the array containing the number of support values to remove for each var
    CT table=readFile(argv[1]);

    //we initialize the solver specific data
    sData.deltaXs=(int **) malloc(sizeof(int*)*table.variablesNo);
    sData.deltaXSizes=(int*) malloc(sizeof(int)*table.variablesNo);
    sData.domainSizes=(int*) malloc(sizeof(int)*table.variablesNo);
    sData.domains=(char**) malloc(sizeof(char*)*table.variablesNo);
    for (int i = 0; i < table.variablesNo; i++){
    	sData.deltaXs[i]=(int*) malloc(sizeof(int)*maxValuesToRemove);
    	sData.domains[i]=(char*) malloc(sizeof(char)*table.supportSizes[i]);
    	sData.domainSizes[i]=table.supportSizes[i];
    	sData.deltaXSizes[i]=0;
    }

    //we initialzie the number of iterations and at each we prune some elements of the supports
    int iterations=1;
    randomSolve(&table,iterations);


    //free memory
    free(sData.deltaXSizes);
    for (int i = 0; i < table.variablesNo; i++){
    	free(sData.deltaXs[i]);
    	free(sData.domains[i]);
    }
    free(sData.domains);
    free(sData.deltaXs);
}