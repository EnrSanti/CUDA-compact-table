#include "table.hpp"
#include <unistd.h>
#include <chrono>
//#define RECORD_OUTPUT

Table::Table(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) :
    Constraint(vars[0]->getSolver()), 
    _vars(vars), _tuples(tuples), 
    _currTable(SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size())){
    
    
    auto start = std::chrono::high_resolution_clock::now();

    //auto start = std::chrono::high_resolution_clock::now();
    int noTuples=tuples.size();
    int noVars=vars.size();
    
    _s_val= vector<int>();
    _s_sup= vector<int>();
    _supportOffsetJmp=vector<int>(noVars);
    _variablesOffsets=vector<int>(noVars);


    _currTable=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size());

    for (int i = 0; i < noVars; i++){        
        //calculating the number of rows in the support bitset
        _supportSize+=vars[i]->intialSize();
        //store the offset
        _variablesOffsets[i]=vars[i]->initialMin();      
    }


    //calculating the offset of the variables, used in accessing the support rows    
    _supportOffsetJmp[0]=0;
    for (int i = 1; i < noVars; i++){
        _supportOffsetJmp[i]=_supportOffsetJmp[i-1]+vars[i-1]->intialSize();
    }

    //we allocate and initialize the support bitsets
    currTableSize=(noTuples/32)+1; 
    _supports=(unsigned int*) calloc(_supportSize*currTableSize,sizeof(unsigned int));

    
    bool found=false;
    int tuplesOfSingletons[noVars];

    for (int v = 0; v < noVars; v++){
        tuplesOfSingletons[v]=-1;
        for (int t = 0; t < noTuples; t++){
            if(tuples[t][v]>=_vars[v]->initialMin() && tuples[t][v]<=_vars[v]->initialMax()){
                //classical entry (we update all the supports in the same way)
                int entryValue=tuples[t][v]-_variablesOffsets[v];   
                
                int offset=(_supportOffsetJmp[v]+entryValue)*currTableSize;

                addToMaskInt(&(_supports[offset]),t+1); 

                found=true;
                tuplesOfSingletons[v]=t;
            }else{
                _currTable.addToMaskInt(t+1);
            }
        }
        if (found==false){
            failNow();
            return;
        }
    }
    
    _currTable.reverseMask();
    _currTable.intersectWithMask();
    _currTable.clearMask();

    //forall vars
    for (int i = 0; i < noVars; i++){
        if(_vars[i]->size()==1){
            if(tuplesOfSingletons[i]==-1){
                failNow();
                return;
            }
        }
    }
    if(_currTable.isEmpty()){
        failNow();
        return;
    }

    #ifdef RECORD_OUTPUT
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        printf("%%%%%% Time to init table (serial): %ld us\n", duration.count());
    #endif
}

void Table::post()
{
    propagate();
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }

}

void Table::propagate()
{
    enfoceGAC();

    
}




//---------------------------------------------
//------- The three functions of alg. 2 -------
//---------------------------------------------

void Table::updateTable(){
    //forall var x in s_val
    int index=0;
  
    

    for(int i=0; i < _s_val.size(); ++i){
        _currTable.clearMask();
        index=_s_val[i];

        
        //reset based update
        for (int j = _vars[index]->min(); j <= _vars[index]->max();  j++){ 

            if(_vars[index]->contains(j)){

                int index_x_a=(_supportOffsetJmp[index]+j-_variablesOffsets[index])*currTableSize;
                _currTable.addToMaskArray(&(_supports[index_x_a]));
         
            }
            

        } 

        _currTable.intersectWithMask();

        //printf("%%%%%% ct after %d: ",index); 
        //for(int k=0; k<currTableSize; k++)
        //    printf(" %d ",_currTable._words[k].value());
        //printf("\n");
    }
 
    
    
    if(_currTable.isEmpty()){
        failNow();
        return;
    }
}

void Table::filterDomains(){


    for(int i=0; i < _s_sup.size(); ++i){
        int index=_s_sup[i];
        for (int j = _vars[index]->min(); j <= _vars[index]->max(); j++){
            if(_vars[index]->contains(j)){ //i.e. a \in dom(x)

                int index_x_a=_supportOffsetJmp[index]+j-_vars[index]->initialMin();
                int indexResidue=intersectIndexSparse(&_supports[index_x_a*currTableSize],_currTable);
                    
                if(indexResidue==-1){
                    _vars[index]->remove(j);        
                }     
            }
        }
    }
}

void Table::enfoceGAC(){

    auto start = std::chrono::high_resolution_clock::now();
    _s_val.clear();
    _s_sup.clear();
    _s_val.shrink_to_fit();
    _s_sup.shrink_to_fit();
	for (int i = 0; i < _vars.size(); i++){
		
        if(_vars[i]->changed()){
            _s_val.push_back(i);
        }
        
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
        }
	}
    
	auto startUpdate = std::chrono::high_resolution_clock::now();
    updateTable();
    auto endUpdate = std::chrono::high_resolution_clock::now();
	
    auto startFilter = std::chrono::high_resolution_clock::now();
    filterDomains();
    auto endFilter = std::chrono::high_resolution_clock::now();
    
    #ifdef RECORD_OUTPUT
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        auto durationUpdate = std::chrono::duration_cast<std::chrono::microseconds>(endUpdate - startUpdate);
        auto durationFilter = std::chrono::duration_cast<std::chrono::microseconds>(endFilter - startFilter);
        printf("%%%%%% Time to propagate: %ld us (update %ld) (filter %ld)\n", duration.count(), durationUpdate.count(), durationFilter.count());
    #endif
    

}

void Table::addToMaskInt(unsigned int* mask,int value){  
	int offset;
    int bitsPerWord=32;
	unsigned int wordToOr=(unsigned int) 1<<(bitsPerWord-(value%bitsPerWord));
   
	int wordIndex=floor(value/bitsPerWord);
	if(value%bitsPerWord==0){
		wordIndex--;
	}
	mask[wordIndex]=mask[wordIndex] | wordToOr;
}
int Table::intersectIndexSparse(unsigned int* words,SparseBitSet& m) {

   int offset;
   for (int i = 0; i < currTableSize; i++) {
   
      if ((words[i] & m[i]) != 0)
         return i;
   }
   return -1;
}
