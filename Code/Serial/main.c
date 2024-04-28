#include <stdio.h>
#include "constraintCT.c"
#include <math.h>

//-----------------------------------------------------------------------
//This file provides a naive "solver" simulator which keeps track of the variables
//and the domains, and at each iteration (no. fixed by "iterations") removes 
//some random values from the domains
//-----------------------------------------------------------------------

//data to random remove supports:

int seed=12;
int maxValuesToRemove=2; //at each iteration how many values do we remove from a SINGLE variable
int iterations=4; //the number of iterations the simulator will do
solverData sData; //the data related to the "solver simulator (i.e. domains, vars..)"

//given a (postive) table we return whether a solution may still be found
int hasSolution(CT table){
	return !isEmpty((table.currTable));
}

//given a (postive) table and a value, we generate a number of values to remove from each variable
int* genNoSupportsVals(CT *table, int maxValuesToRemove){
	for (int i = 0; i < table->variablesNo; i++) {
		//if maxValuesToRemove is bigger than the current domain size we can't remove that many ponits 
		int lowerbound= (maxValuesToRemove < table->lastSizes[i]) ? maxValuesToRemove : table->lastSizes[i];
        sData.deltaXSizes[i] = rand() % (lowerbound) + 1; //random no between 1 and lowerbound
    }
}

//prints the values in deltaXs (i.e. for each variable which values are removeds)
void printValsToRemove(CT *table){
	printf("\n----------------------------------------------------------------------\n----------------------------------------------------------------------\n");
	for (int i = 0; i < table->variablesNo; i++){
		printf("Remove from %s: ",table->scope[i]);
		for (int j = 0; j < sData.deltaXSizes[i]; j++){
			printf("%d, ",sData.deltaXs[i][j]);
		}
		if(sData.deltaXSizes[i]==0){
			printf("NONE, ");
		}
		printf("The domain size after: %d",sData.domainSizes[i]);
		printf("\n");
	}
	printf("-----------------------------------\n");
	
}
void printDomains(CT *table){
	for (int i = 0; i < table->variablesNo; ++i){
		printf("%s: ",table->scope[i]);
		for (int j = 0; j < table->supportSizes[i]; ++j){
			printf("%d,",sData.domains[i][j]);
		}
		printf("\n");
	}
}
//given a table it removes some domain values from each variable (up to maxValuesToRemove, for each var.)
void RemoveRandomSupports(CT *table){

	long minValueInSupport;
	long maxValueInSupport;
	int toRemove;

	//for each var we seek the number (random) of values to remove, stores them into sData.deltaXSizes
	genNoSupportsVals(table,maxValuesToRemove); 
	printf("\nfor the next iteration:\n");
	//for each var we get some (deltaXSizes[i]) random vars in the doman and remove them
	for (int i = 0; i < table->variablesNo; i++){
		
		minValueInSupport=table->variablesOffsets[i];
		maxValueInSupport=minValueInSupport+table->supportSizes[i]-1;
		toRemove=sData.deltaXSizes[i];

		printf("number of vals to remove from %s: %d\n",table->scope[i],toRemove);
		for (int j = 0; j < toRemove; j++){
			//we now generate the actual values to remove
			sData.deltaXs[i][j]=(rand() % (maxValueInSupport - minValueInSupport + 1)) + minValueInSupport; //random no between min and max
			printf("we wish to remove from %s: %d\n",table->scope[i],sData.deltaXs[i][j]);
			//some values may not be in the domain so we don't actually decrease delta
			if(sData.domains[i][sData.deltaXs[i][j]-table->variablesOffsets[i]]==0)
				sData.deltaXSizes[i]--;
			sData.domains[i][sData.deltaXs[i][j]-table->variablesOffsets[i]]=0;	
		}
		sData.domainSizes[i]-=sData.deltaXSizes[i];
	}
	
	printValsToRemove(table);
}

//the main procedure invoekd by the main, we loop "iterations" times
void randomSolve(CT *table,int iterations){
	for (int i = 0; i < iterations; i++)	{
		
		RemoveRandomSupports(table);
		enfoceGAC(table,&sData); //we discard for now the return value (backtrack, todo: use it)
		printf("\nafter GAC (the domains):\n");
		printDomains(table);
		
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
    printf("\nDomains:\n");
	printDomains(&table);
    //at each iteration we prune some elements of the supports
    randomSolve(&table,iterations);



    //free memory:
    free(sData.deltaXSizes);
    for (int i = 0; i < table.variablesNo; i++){
    	free(sData.deltaXs[i]);
    	free(sData.domains[i]);
    }
    free(sData.domains);
    free(sData.deltaXs);
}

