#include <stdio.h>
#include "constraintCT.c"
#include <math.h>

int seed=123;
int maxValuesToRemove=3;
int** deltaXs;
int* deltaXSizes;


int hasSolution(CT table){
	printf("%d",isEmpty((table.currTable)));
	return 2;
}

int* genNoSupportsVals(CT *table, int maxValuesToRemove){
	int lowerbound= (maxValuesToRemove < table->variablesNo) ? maxValuesToRemove : table->variablesNo;
	for (int i = 0; i < table->variablesNo; i++) {
        deltaXSizes[i] = rand() % (lowerbound + 1); //random no between 0 and maxValuesToRemove
    }
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
		for (int j = 0; j < deltaXSizes[i]; j++){
			//we now generate the actual values to remove
			deltaXs[i][j]=(rand() % (maxValueInSupport - minValueInSupport + 1)) + minValueInSupport; //random no between min and max
			//todo 1 volta funziona poi no (ovviamente)
		}
		//printf("for var %d i can remove %ld...%ld\n",i, minValueInSupport,maxValueInSupport);
	}
}

void randomSolve(CT table,int iterations){
	for (int i = 0; i < iterations; i++)	{
		if(!hasSolution(table)){
			printf("\n--------------- The problem has no solution ---------------\n");
			return;
		}
		RemoveRandomSupports(&table);
		int enfoceGAC(&table)
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
    deltaXs=(int **) malloc(sizeof(int*)*table.variablesNo);
    deltaXSizes=(int*) malloc(sizeof(int)*table.variablesNo);
    for (int i = 0; i < table.variablesNo; i++){
    	deltaXs[i]=(int*) malloc(sizeof(int)*maxValuesToRemove);
    }

    //we initialzie the number of iterations and at each we prune some elements of the supports
    int iterations=1;
    randomSolve(table,iterations);


    //free memory
    free(deltaXSizes);
    for (int i = 0; i < table.variablesNo; i++){
    	free(deltaXs[i]);
    }
}