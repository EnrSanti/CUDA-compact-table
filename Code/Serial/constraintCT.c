#include <stdio.h>
#include "../Shared/shared.h"


CT data;

//---------------------------------------------
//---- Aux functions to access the support ----
//---------------------------------------------

//eg which is the index to access support[x,a]
int getSupportIndex(CT *table,int var, int value){
	return table->supportOffsetJmp[var]+(value-(table->variablesOffsets[var]));
}


//---------------------------------------------
//------- The three functions of alg. 2 -------
//---------------------------------------------

void updateTable(CT *data,int** deltaXs,int* deltaXSizes){
	//for now we loop through all the array, with a list we could be more efficient (todo)
	for (int i=0; i<data->variablesNo; i++){
		if(data->s_val[i]==1){
			clearMask(&(data->currTable));
			if(deltaXSizes[i]<data->lastSizes[i]){
				for (int j = 0; j < deltaXSizes[i]; j++){
					int index=getSupportIndex(data,i,deltaXs[i][j]);
					addToMask(&(data->currTable),data->supports[index].words); 		
				}
				reverseMask(&(data->currTable));
			}else{
				//to improve with lists: todo
				for (int j = 0; j < data->supportSizes[i]; j++){
					int index=getSupportIndex(data,i,j+data->variablesOffsets[i]);
					addToMask(&(data->currTable),data->supports[index].words); 		
				}
			}
			intersectWithMask(&(data->currTable));
			if(isEmpty(data->currTable))
				break;
		}
	}

}
void filterDomains(CT *data,char** domains,int* domainSizes){
	//for now we loop through all the array, with a list we could be more efficient (todo)
	for (int i = 0; i < data->variablesNo; i++){
		if(data->s_sup[i]==1){
			//todo for all values in dom x, che si possono prendere dal processo di search (for now we loop through all the array todo: lists)
			for (int j = 0; j < data->supportSizes[i]; j++){
				int x_aIndex=getSupportIndex(data,i,j+data->variablesOffsets[i]);
				int index=data->residues[x_aIndex];
				if((data->currTable.words[index] & (data->supports[x_aIndex]).words[index] ) == 0x0000000000000000){
					index=intersectIndex(&(data->currTable),data->supports[x_aIndex].words);
					if(index!=-1){
						data->residues[x_aIndex]=index;
					}else{
						if(domains[i][j-data->variablesOffsets[i]]!=0)
							domainSizes[i]-=1;
						domains[i][j-data->variablesOffsets[i]]=0;
					}

				}
			}
			data->lastSizes[i];//=dom(x) todo;
		}
	}
}
int enfoceGAC(CT *data,solverData *sData){
	//this for correspond to the assignment of s_val
	int aux=0;


	for (int i = 0; i < data->variablesNo; i++){
		//update s_val
		data->s_val[i]=(sData->domainSizes[i] != data->lastSizes[i]) ? 1 : 0;

		if(data->s_val[i]==1){
			data->lastSizes[i]=sData->domainSizes[i];
		}
		//update s_sup
		data->s_sup[i]=(sData->domainSizes[i]>1) ? 1 : 0;
	}


	updateTable(data,sData->deltaXs, sData->deltaXSizes);
	if(isEmpty(data->currTable)){
		return -1; //backtrack
	}
	filterDomains(data,sData->domains,sData->domainSizes);
	return 0;
}


