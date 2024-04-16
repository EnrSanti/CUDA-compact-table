#include <stdio.h>
#include "constraintCT.c"
#include <math.h>


//data to random remove supports
int seed=124;
int maxValuesToRemove=1;
solverData sData;
int iterations=3;


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
		printf("The domain size after: %d",sData.domainSizes[i]);
		printf("\n");
	}
	printf("-----------------------------------\n");
	
}

void RemoveRandomSupports(CT *table){

	long minValueInSupport;
	long maxValueInSupport;
	int actualRemoved;
	int toRemove;
	//for each var we seek the number of values to remove
	genNoSupportsVals(table,maxValuesToRemove); 

	//for each var we get some (deltaXSizes[i]) random vars in the doman and remove them
	for (int i = 0; i < table->variablesNo; i++){
		
		minValueInSupport=table->variablesOffsets[i];
		maxValueInSupport=minValueInSupport+table->supportSizes[i]-1;
		actualRemoved=sData.deltaXSizes[i];
		for (int j = 0; j < sData.deltaXSizes[i]; j++){
			//we now generate the actual values to remove
			sData.deltaXs[i][j]=(rand() % (maxValueInSupport - minValueInSupport + 1)) + minValueInSupport; //random no between min and max
			if(sData.domains[i][sData.deltaXs[i][j]-table->variablesOffsets[i]]==0){
				actualRemoved--;
			}else{
				sData.domains[i][sData.deltaXs[i][j]-table->variablesOffsets[i]]=0;
			}
		}
		sData.domainSizes[i]-=actualRemoved;
	}
	printf("\nhere:\n");
	for (int i = 0; i < table->variablesNo; ++i){
		for (int j = 0; j < table->supportSizes[i]; ++j){
			printf("%d,",sData.domains[i][j]);
		}
		printf("\n");
	}
	printValsToRemove(table);
}

void randomSolve(CT *table,int iterations){
	for (int i = 0; i < iterations; i++)	{
		
		RemoveRandomSupports(table);
		enfoceGAC(table,&sData); //we discard for now the return value (backtrack, todo: use it)
		printCurrTable(table);
		if(!hasSolution(*table)){
			printf("\n--------------- The problem has no solution ---------------\n");
			return;
		}
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
    	//naive way, still a part of the solver not mine :)
    	for (int j = 0; j < table.supportSizes[i]; j++){
    		sData.domains[i][j]=1;
    	}
    	sData.domainSizes[i]=table.supportSizes[i];
    	sData.deltaXSizes[i]=0;
    }

    //at each iteration we prune some elements of the supports
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