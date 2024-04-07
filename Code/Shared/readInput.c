#include <stdio.h>
#include <stdbool.h>
#include <string.h> 
#include <stdlib.h> 
#include "RSparseBitSet.c"
#include "CTData.c"

CT readFile(const char* str) {

    FILE* ptr;

    //we open the file and check if it exists
    ptr = fopen(str, "r");
    if (NULL == ptr) {
        printf("No such file or can't be opened %s",str);
        exit(1);
    }


    //read and populate the CT fields
    CT data;
    
    //first we get how many variables there are overall
    int noVars=0;
    int noTuples=0;  
    fscanf(ptr, "vars no: %d;\n", &noVars);
    fscanf(ptr, "tuples no: %d;\n", &noTuples);
    
    //we instanciate and populate the number of variables and their names in the CT
    data.variablesNo=noVars;
    data.scope=(char**) malloc(noVars*sizeof(char*));

    //for each var we read the domain and name
    //we temporary save (for each var) the domanin bounds in two arrays
    int* domainMin=(int*) malloc(noVars*sizeof(int));
    int* domainMax=(int*) malloc(noVars*sizeof(int));
    //we also save
    for (int i = 0; i < noVars; i++){
        //var name up to 100 chars, later we store them in scope with their actual size
        char var_name[100];
        fscanf(ptr, "var %d..%d: %[^;]; ", &(domainMin[i]), &(domainMax[i]),var_name);
        
        //copy everything to the scope 
        int len=strlen(var_name);
        data.scope[i]=(char*) malloc((len+1)*sizeof(char));
        strncpy(data.scope[i],var_name,len);
    }


    //go over the "predicate" string in the file
    fscanf(ptr, "predicate table (");   

  
    //we see which variables are actually involved in the table constraint
    char var[100];
    while (fscanf(ptr, "%[^),],", var) == 1) {
        //printf("%s \n",var);
        ;
    }

    //go over the string in the file
    fscanf(ptr, ") {\n"); 


    //we populate the CT
    data.currTable=createBitSet(noTuples,0);
    data.lastSizes=(int*) malloc(noVars*sizeof(int));
    
    long* skipSupportVar=(long*) malloc(noVars*sizeof(long));
    int supportSize=0;
    for (int i = 0; i < noVars; i++){
        data.lastSizes[i]=domainMax[i]-domainMin[i]+1;
        supportSize+=data.lastSizes[i];
    }
    skipSupportVar[0]=0;
    for (int i = 1; i < noVars; i++){
        skipSupportVar[i]=skipSupportVar[i-1]+data.lastSizes[i-1];
    }

    printf("the supports will have size of%d \n",supportSize );
    data.supportSize=supportSize;
    data.supports=(bitSet*) calloc(supportSize,sizeof(bitSet));

 

    //we allocate and initialize the support bitsets
    for (int i = 0; i < supportSize; i++){
        data.supports[i]=createBitSet(noTuples,0); //the content doesn't make sense yet, later we need to update the mask and intersect it
    }

   
    //we read and count the rows of the table
    int ctr=0;
    int constrNo=1;
    int* row=(int*) malloc(noVars*sizeof(int));
    while (fscanf(ptr, "%[^;,],\n", var) == 1) {
        if(strncmp(var,"\n}",2)!=0 && strncmp(var,"}",1)!=0){
            fscanf(ptr, ";\n"); 
            row[ctr]=atoi(var);    
            ctr++;
            if(ctr%noVars==0){
                for (int i = 0; i < noVars; ++i){
                    printf("row: %d %d \n",skipSupportVar[i]+row[i],constrNo);
                    addToMaskInt(&(data.supports[skipSupportVar[i]+row[i]]),constrNo);   
                }

                ctr=0;
                constrNo++;
                //we have read a full row
            }
        }
    }
    

    //then we update the bitsets with the mask to be coherent, lastly we rest the mask
    for (int i = 0; i < supportSize; ++i){  
        intersectWithMask(&(data.supports[i]));
        clearMask(&(data.supports[i]));
    
    }
    
    printCT(&data);
    
    //TODO FIX MEM LEAK
    fclose(ptr);
    return data;
}