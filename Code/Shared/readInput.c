#include <stdio.h>
#include <string.h> 
#include <stdlib.h> 
#include <stdbool.h>
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
    data.variablesOffsets=(long*) malloc(noVars*sizeof(long));

    //for each var we read the domain and name
    //we temporary save (for each var) the domanin bounds in two arrays
    int* domainMin=(int*) malloc(noVars*sizeof(int));
    int* domainMax=(int*) malloc(noVars*sizeof(int));

    //we also save
    for (int i = 0; i < noVars; i++){
        //var name up to 100 chars, later we store them in scope with their actual size
        char var_name[100];
        fscanf(ptr, "var %d..%d: %[^;]; ", &(domainMin[i]), &(domainMax[i]),var_name);
        
        //we store the offset
        data.variablesOffsets[i]=domainMin[i];
        //copy name to the scope 
        int len=strlen(var_name);
        data.scope[i]=(char*) malloc((len+1)*sizeof(char));
        strncpy(data.scope[i],var_name,len);
    }


    //go over the "predicate" string in the file
    fscanf(ptr, "predicate table (");   

  
    //we see which variables are actually involved in the table constraint (for now all of them)
    char var[100];
    while (fscanf(ptr, "%[^),],", var) == 1) {
        //printf("%s \n",var);
        ;
    }

    //go over the string in the file
    fscanf(ptr, ") {\n"); 


    //we populate the CT
    data.currTable=createBitSet(noTuples);
    data.lastSizes=(int*) malloc(noVars*sizeof(int));
    data.supportSizes=(long*) malloc(noVars*sizeof(long));
    data.supportOffsetJmp=(long*) malloc(noVars*sizeof(long));
    data.s_val=(int *)malloc(noVars*sizeof(int));
    data.s_sup=(int *)malloc(noVars*sizeof(int));

    int supportSize=0;
    for (int i = 0; i < noVars; i++){
        //in the mean time we intialize
        data.s_val[i]=0;
        data.s_sup[i]=0;

        data.lastSizes[i]=domainMax[i]-domainMin[i]+1;
        data.supportSizes[i]=(long)data.lastSizes[i];
        supportSize+=data.lastSizes[i];
    }

    data.supportOffsetJmp[0]=0;
    for (int i = 1; i < noVars; i++){
        data.supportOffsetJmp[i]=data.supportOffsetJmp[i-1]+data.lastSizes[i-1];
    }

    printf("the supports will have size (rows): %d \n",supportSize);
    data.supportSize=supportSize;
    data.residues= (long*) malloc(supportSize*sizeof(long));
    data.supports=(bitSet*) malloc(supportSize*sizeof(bitSet));
    //we initialize also the additional support for short values
    data.supportsShort=(bitSet*) malloc(supportSize*sizeof(bitSet));
    //and for smart tables
    data.supportsMin=(bitSet*) malloc(supportSize*sizeof(bitSet));
    data.supportsMax=(bitSet*) malloc(supportSize*sizeof(bitSet));
 
    //we allocate and initialize the support bitsets
    for (int i = 0; i < supportSize; i++){
        data.supports[i]=createBitSet(noTuples); //the content doesn't make sense yet, later we need to update the mask and intersect it
        data.supportsShort[i]=createBitSet(noTuples); 
        data.supportsMax[i]=createBitSet(noTuples); 
        data.supportsMin[i]=createBitSet(noTuples); 
    }

    //we read and count the rows of the table
    int ctr=0;
    int constrNo=1; //keeps track of the rows
    int offset;

    while (fscanf(ptr, "%[^;,],\n", var) == 1) {
        if(strncmp(var,"\n}",2)!=0 && strncmp(var,"}",1)!=0){ //check if I have read the last } of the table (may be after a new line)

            fscanf(ptr, ";\n");
            //warning: code duplication, but this is just for testing 

            //we check if it's not a short entry (i.e. has *)
            if(strncmp(var,"*",1)==0){
                //supports, supportMin, supportMax we need to set all the var values to 1
                //done
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);  
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);   
                }
                //supportsShort to 0, so we don't add anything :)

            }else if(strncmp(var,"!=",2)==0){ //if we have a smart entry !=
                //done
                int entryValue=atoi(var+2)-data.variablesOffsets[ctr]; //we skip over the chars != (no spaces allowed), -varOffset, to retrieve the index of the value on it's domain, i.e. x 4..10, if we write 6 we mean the second possible value of x 
                for (int varValue = 0; varValue < entryValue; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);   
                }
                for (int varValue = entryValue+1; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);   
                }
                //TODO CHECK IF THE LAST VALUE HAS TO BE INCLUDED (i.e. x!=c for supportMin or x!=a for supportMax)
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);  
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);   
                }
                // supportsShort to 0, so we don't add anything :)
            
            }else if(strncmp(var,"<=",2)==0){

                //done
                int entryValue=atoi(var+2)-data.variablesOffsets[ctr];

                for (int varValue = 0; varValue <= entryValue; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);
                }

                //set supportsMax all to 1
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);   
                }
                // supportsShort to 0, so we don't add anything :)

            }else if(strncmp(var,"<",1)==0){
                //done
                int entryValue=atoi(var+1)-data.variablesOffsets[ctr];


                for (int varValue = 0; varValue < entryValue; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);
                }
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);   
                }
                // supportsShort to 0, so we don't add anything :)
            }else if(strncmp(var,">=",2)==0){
                //done
                int entryValue=atoi(var+2)-data.variablesOffsets[ctr];

                for (int varValue = entryValue; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);
                }       
                //set supportsMin all to 1
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMax
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);   
                }

                // supportsShort to 0, so we don't add anything :)
            }else if(strncmp(var,">",1)==0){
                int entryValue=atoi(var+1)-data.variablesOffsets[ctr];
                for (int varValue = entryValue+1; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supports[offset]),constrNo);  
                    //update supportsMax
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);
                }  
                for (int varValue = 0; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);   
                }
                // supportsShort to 0, so we don't add anything :)
            }else{ //classical entry (we update all the supports in the same way)
                //done
                int entryValue=atoi(var)-data.variablesOffsets[ctr];   
                offset=data.supportOffsetJmp[ctr]+entryValue;
                addToMaskInt(&(data.supports[offset]),constrNo);   
                addToMaskInt(&(data.supportsShort[offset]),constrNo); 


                for (int varValue = 0; varValue <= entryValue; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    //update supportsMin
                    addToMaskInt(&(data.supportsMin[offset]),constrNo);   
                }
                for (int varValue = entryValue; varValue < data.supportSizes[ctr]; varValue++) {
                    offset=data.supportOffsetJmp[ctr]+varValue;
                    addToMaskInt(&(data.supportsMax[offset]),constrNo);   
                }
            }

            ctr++;

            if(ctr%noVars==0){
                ctr=0;
                constrNo++;
                //we have read a full row
            }
        }
    }

    //then we update the bitsets with the mask to be coherent, lastly we rest the mask
    for (int i = 0; i < supportSize; ++i){  
        intersectWithMask(&(data.supports[i]));
        intersectWithMask(&(data.supportsShort[i]));
        intersectWithMask(&(data.supportsMin[i]));
        intersectWithMask(&(data.supportsMax[i]));
        clearMask(&(data.supports[i])); 
        clearMask(&(data.supportsShort[i])); 
        clearMask(&(data.supportsMin[i])); 
        clearMask(&(data.supportsMax[i])); 
        //we initialize residues
        for(int j=0; j<noTuples; j++){
            if(data.supports[i].words[j/bitsPerWord]!=0x0000000000000000){
                data.residues[i]=j; 
                break;
            }
        }
    }
    
    printCTData(&data);
    
    //we free the memory used
    free(domainMin);
    free(domainMax);
    

    //close the file and return the ct structure
    fclose(ptr);
    return data;
}