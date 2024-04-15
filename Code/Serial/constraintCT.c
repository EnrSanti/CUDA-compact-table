#include <stdio.h>
#include "../Shared/shared.h"


CT data;



void updateTable(CT *data){
	for (int i=0; i<data->variablesNo; i++){
		int delta_x=data->prevLastSizes[i]-data->lastSizes[i];
		if(delta_x!=0){ //is the same as if variable in S_val
			clearMask(&(data->currTable));
			if(delta_x<data->lastSizes[i]){
				//todo	//i valori di delta_x si possono prendere dal processo di search 
				;
			}else{
				//todo //i valori di delta_x si possono prendere dal processo di search 
				;
			}
			intersectWithMask(&(data->currTable));
			if(isEmpty(data->currTable))
				break;
			
		}
	}

}
void filterDomains(CT *data){
	for (int i = 0; i < data->variablesNo; ++i){
		if(data->s_sup[i]==1){
			//todo for all values in dom x, che si possono prendere dal processo di search
		}
	}
}
int enfoceGAC(CT *data){
	//this for correspond to the assignment of s_val
	int aux=0;
	for (int i = 0; i < data->variablesNo; ++i){
		int aux=data->lastSizes[i];
		if(data->prevLastSizes-data->lastSizes[i]!=0){
			data->prevLastSizes[i]=data->lastSizes[i];
			//data->lastSizes[i]= //todo //i valori di delta_x si possono prendere dal processo di search 
		}
	}

	//i valori di delta_x si possono prendere dal processo di search 
	//todo for
	updateTable(data);
	if(isEmpty(data->currTable)){
		return -1; //backtrack
	}
	filterDomains(data);
	return 0;
}