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

void updateTable(CT *data,int** deltaXs,int* deltaXSizes, int* domainSizes, char** domains){
	//for now we loop through all the array, with a list we could be more efficient (todo)
	for (int i=0; i<data->variablesNo; i++){
		if(data->s_val[i]==1){
			clearMask(&(data->currTable));

			if(deltaXSizes[i]+2<domainSizes[i] && 1==0){ //TODO TOGLI SECONDA CONDIZIONE, in questa versione incremental update non è considerato
				//printf("\n++++incremental update++++\n");
				int domMin=-1; //TODO CAMBIA
				int domMax=-1; //TODO CAMBIA
				for (int j = 0; j < deltaXSizes[i]; j++){
					if(deltaXs[i][j]>domMin && deltaXs[i][j]<domMax){
						int index=getSupportIndex(data,i,deltaXs[i][j]);
						addToMask(&(data->currTable),data->supportsShort[index].words); 		
					}
				}
				reverseMask(&(data->currTable));
				/* TODO UNCOMMENT
				if(dom(i).minChanged()){
					...	
				}
				if(dom(i).maxChanged()){
					...
				}
				*/
			}else{
				//printf("\n++++reset based update++++\n"); 
				for (int j = 0; j < data->supportSizes[i]; j++){
					if(domains[i][j]==1){
						int index=getSupportIndex(data,i,j+data->variablesOffsets[i]);
						addToMask(&(data->currTable),data->supports[index].words); 		
					}
				}
			}
			intersectWithMask(&(data->currTable));
			if(isEmpty(data->currTable)){
				return;
			}
		}
	}

}
void filterDomains(CT *data,char** domains,int* domainSizes){
	//for now we loop through all the array, with a list we could be more efficient (todo)
	for (int i = 0; i < data->variablesNo; i++){
		if(data->s_sup[i]==1){
			//for all values in dom x, che si possono prendere dal processo di search (for now we loop through all the array todo: lists)
			for (int j = 0; j < data->supportSizes[i]; j++){
				if(domains[i][j]==1){ //i.e. a \in dom(x)
					int x_aIndex=getSupportIndex(data,i,j+data->variablesOffsets[i]);
					int index=data->residues[x_aIndex];
					if((data->currTable.words[index] & (data->supports[x_aIndex]).words[index] ) == 0x0000000000000000){
						index=intersectIndex(&(data->currTable),data->supports[x_aIndex].words);
						if(index!=-1){
							data->residues[x_aIndex]=index;
						}else{
							if(domains[i][j]!=0)
								domainSizes[i]-=1;
							domains[i][j]=0;

						}

					}
				}
			}
			data->lastSizes[i]=domainSizes[i];//=dom(x) todo;
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

	updateTable(data,sData->deltaXs,sData->deltaXSizes,sData->domainSizes,sData->domains);
	
	if(isEmpty(data->currTable)){
		return -1; //backtrack
	}
	filterDomains(data,sData->domains,sData->domainSizes);
	
	return 0;
}

