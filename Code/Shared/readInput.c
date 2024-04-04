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
    
    //first we get how many variables there are overall (which can be more than the ones in a table)
    int noVars=0;
    int noTuples=0;  
    fscanf(ptr, "vars no: %d;\n", &noVars);
    fscanf(ptr, "tuples no: %d;\n", &noTuples);
    
    //we instanciate and populate the CT
    data.scope=(char**) malloc(noVars*sizeof(char*));

    //for each var we read the domain and name
    //we temporary save (for each var) the domanin bounds in two arrays
    int* domainMin=(int*) malloc(noVars*sizeof(int));
    int* domainMax=(int*) malloc(noVars*sizeof(int));
    
    for (int i = 0; i < noVars; i++){
        //var name up to 100 chars, later we store them in scope with their actual size
        char var_name[100];
        fscanf(ptr, "var %d..%d: %[^;]; ", &(domainMin[i]), &(domainMax[i]),var_name);
        printf("var %d..%d: %s;\n", domainMin[i], domainMax[i], var_name);
        
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
        printf("%s \n",var);
    }

    //go over the string in the file
    fscanf(ptr, ") {\n"); 




    //we instanciate and populate the CT
    data.currTable=createBitSet(noTuples,0);
    data.lastSizes=(int*) malloc(noVars*sizeof(int*));
    
    int supportSize=0;
    for (int i = 0; i < noVars; i++){
        data.lastSizes[i]=domainMax[i]-domainMin[i]+1;
        supportSize+=data.lastSizes[i];
    }

    printf("the supports will have size of%d \n",supportSize );
    data.supports=(bitSet*) malloc(supportSize*sizeof(bitSet*));

    for (int i = 0; i < supportSize; ++i){
        data.supports[i]=createBitSet(noVars,0); //the content doesn't make sense yet, later we need to update the mask and intersect it
    }

    //we read and count the rows of the table
  
    int ctr=0;
    int* row=(int*) malloc(noVars*sizeof(int));;
    while (fscanf(ptr, "%[^;,],\n", var) == 1) {
        if(strncmp(var,"\n}",2)!=0 && strncmp(var,"}",1)!=0){
            fscanf(ptr, ";\n"); 
            printf("%s\n",var);
            row[ctr]=atoi(var);    
            ctr++;
            if(ctr%noVars==0){
                ctr=0;
                //addToMask(); //TODO
                
                //we have read a row
            }
        }
    }
    printf("no noTuples %d\n",noTuples);

    
    for (int i = 0; i < noTuples; ++i){
        intersectWithMask(&(data.supports[i]));
        clearMask(&(data.supports[i]));
    }

    

    fclose(ptr);
    return data;
}