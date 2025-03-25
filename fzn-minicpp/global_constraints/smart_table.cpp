#include "smart_table.hpp"
#include "chrono"
enum SmartTableOp {Eq=1, All=2 ,LtInt=3, GtInt=5 /*, LtVar=4 /*, GtVar=6*/};

SmartTable::SmartTable(vector<var<int>::Ptr> & vars,  vector<vector<int>> & tuples, vector<vector<int>> & signs) :
    Constraint(vars[0]->getSolver()), _vars(vars), _tuples(tuples), _signs(signs), _currTable(SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size())){

    
    noTuples=tuples.size();
    int noVars=vars.size();
    
    _s_val= vector<int>();
    _s_sup= vector<int>();
    _supportOffsetJmp=vector<int>(noVars);
    _variablesOffsets=vector<int>(noVars);

    _currTable=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size());

    for (int i = 0; i < noVars; i++){      

        //calculating the number of rows in the support bitset
        _supportSize+=vars[i]->intialSize();
        //we store the offset
        _variablesOffsets[i]=vars[i]->initialMin();      
        //vars[i]->dumpInSparseBitSet(i,_variablesOffsets[i],vars[i]->min(),vars[i]->initialMin(),vars[i]->max(),_lastVarsValues[i]);

    }


    //calculating the offset of the variables, used in accessing the support rows    
    _supportOffsetJmp[0]=0;
    for (int i = 1; i < noVars; i++){
        _supportOffsetJmp[i]=_supportOffsetJmp[i-1]+vars[i-1]->intialSize();
    }

    //we allocate and initialize the support bitsets
    currTableSize=(noTuples/32)+1; 
    _supports=(unsigned int*) calloc(_supportSize*currTableSize,sizeof(unsigned int));


    intializeTable(noVars,noTuples);

}
void SmartTable::intializeTable(int noVars,int noTuples){

    bool found=false;
    int tuplesOfSingletons[noVars];
    for (int v = 0; v < noVars; v++){

        tuplesOfSingletons[v]=-1;
        for (int t = 0; t < noTuples; t++){
            //if we have a * or an entry that is in the domain of the variable, we need to update the supports
            
            switch(_signs[t][v]) {
                //* entry (we update all the supports in the same way)                
                case SmartTableOp::All:{
                    //set all the bits for the variable in support
                    int offset=_supportOffsetJmp[v]*currTableSize;
                    for(int i=0; i<_vars[v]->intialSize(); i++){
                        addToMaskInt(&(_supports[offset+i]),t+1);
                    }
                    found=true;
                    tuplesOfSingletons[v]=t;
                    break;
                }
                case SmartTableOp::Eq:{
                    
                    if(_vars[v]->contains(_tuples[t][v])){

                        int entryValue=_tuples[t][v]-_variablesOffsets[v];   
                        int offset=(_supportOffsetJmp[v]+entryValue)*currTableSize;
                        //if it's not a *, add one bit to both supports
                        addToMaskInt(&(_supports[offset]),t+1);                       
                        found=true;
                        tuplesOfSingletons[v]=t;
                        
                    }else{
                        _currTable.addToMaskInt(t+1);   
                    }
                    break;
                }


                case SmartTableOp::LtInt:{
                    
                    //populate the supports
                    if(_tuples[t][v]>_vars[v]->initialMin()){
                        
                        int entryValue=_tuples[t][v]-_variablesOffsets[v];   //value of the entry-initial min  
                        if(entryValue>_vars[v]->initialMax())
                            entryValue=_vars[v]->initialMax();
                        int offset=_supportOffsetJmp[v]*currTableSize; //starting point of the supports 
                      
                        
                            
                        for(int i=0; i<entryValue; i++){

                           addToMaskInt(&(_supports[offset+i]),t+1);
                            //don't set anything for supportsShort
                        }

                        //if the value is not the minimum we need to set the bits in supportsMin
                       
                        found=true;
                        tuplesOfSingletons[v]=t;
                        
                    }else{
                        _currTable.addToMaskInt(t+1);   
                    }
                    break;
                }
                //SmartTableOp::GtInt
                case SmartTableOp::GtInt:{
                    if(_tuples[t][v]<_vars[v]->initialMax()){
                        //populate the supports
                        int entryValue=_tuples[t][v]-_variablesOffsets[v];   
                        //if i have x [28,50] but i have a constraint x>30, i need to set the supports from 31 to 50
                        if(entryValue<0)
                            entryValue=-1;
                        int offset=_supportOffsetJmp[v]*currTableSize;
                        
                        for(int i=entryValue+1; i<_vars[v]->intialSize()-1; i++){
                            
                            addToMaskInt(&(_supports[offset+i]),t+1);
                        }
                   
                        found=true;
                        tuplesOfSingletons[v]=t;
                        
                    }else{
                        _currTable.addToMaskInt(t+1);   
                    }
                    break;
                }
                
            }
        }
        if (found==false){
            //printf("%%%%%% EMPTY DOMAIN\n");
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
}
void SmartTable::post(){
    propagate();
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}

void SmartTable::propagate(){
    //auto t0 = std::chrono::high_resolution_clock::now();
    enfoceGAC();
    //auto t1 = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    //printf("%%%%%% Time taken enfGAC serial: %ld microseconds\n", duration.count());
    //fflush(stdout);
}



//---------------------------------------------
//------- The three functions of alg. 2 -------
//---------------------------------------------
void SmartTable::updateTable(){
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

        if(_currTable.isEmpty()){
            failNow();
            return;
		}
    }

}

void SmartTable::filterDomains(){


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
void SmartTable::enfoceGAC(){
    //update the table
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
    
	updateTable();
    //safe
	filterDomains();
    
}
void SmartTable::addToMaskInt(unsigned int* mask,int value){  
	int offset;
    int bitsPerWord=32;
	unsigned int wordToOr=(unsigned int) 1<<(bitsPerWord-(value%bitsPerWord));
   
	int wordIndex=floor(value/bitsPerWord);
	if(value%bitsPerWord==0){
		wordIndex--;
	}
	mask[wordIndex]=mask[wordIndex] | wordToOr;
}
int SmartTable::intersectIndexSparse(unsigned int* words,SparseBitSet& m) {

   int offset;
   for (int i = 0; i < currTableSize; i++) {
   
      if ((words[i] & m[i]) != 0)
         return i;
   }
   return -1;
}