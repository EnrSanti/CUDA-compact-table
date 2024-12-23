#include "table.hpp"
#include <unistd.h>
Table::Table(vector<var<int>::Ptr> & vars, vector<vector<int>> & tuples) :
    Constraint(vars[0]->getSolver()), 
    _vars(vars), _tuples(tuples), 
    _currTable(SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size())){
    
    
    int noTuples=tuples.size();
    int noVars=vars.size();
    
    _s_val= vector<int>();
    _s_sup= vector<int>();
    _supportOffsetJmp=vector<int>(noVars);
    _variablesOffsets=vector<int>(noVars);


    _currTable=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),tuples.size());

    for (int i = 0; i < noVars; i++){        
        //_deltaXs[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),_vars[i]->max()+1);
        //_lastVarsValues[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),_vars[i]->max()+1);

        //calculating the number of rows in the support bitset
        _supportSize+=vars[i]->intialSize();
        //we store the offset
        _variablesOffsets[i]=vars[i]->min();      
        //vars[i]->dumpInSparseBitSet(i,_variablesOffsets[i],vars[i]->min(),vars[i]->initialMin(),vars[i]->max(),_lastVarsValues[i]);

    }


    //calculating the offset of the variables, used in accessing the support rows    
    _supportOffsetJmp[0]=0;
    for (int i = 1; i < noVars; i++){
        _supportOffsetJmp[i]=_supportOffsetJmp[i-1]+vars[i-1]->size();
    }

    //we allocate and initialize the support bitsets
    currTableSize=(noTuples/32)+1; 
    _supports=(unsigned int*) malloc(sizeof(unsigned int)*_supportSize*currTableSize);
    //check allocation
    
    _residues= vector<trail<int>>(_supportSize);


    //we allocate and initialize the support bitsets
    for (int i = 0; i < _supportSize*currTableSize; i++){
        _supports[i]=0x00000000;
    }

    
    
    bool found=false;
    int tuplesOfSingletons[noVars];

    for (int v = 0; v < noVars; v++){
        tuplesOfSingletons[v]=-1;
        for (int t = 0; t < noTuples; t++){
            if(tuples[t][v]>=_vars[v]->min() && tuples[t][v]<=_vars[v]->max()){
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
            //printf("%%%%%% EMPTY DOMAIN\n");
            failNow();
            return;
        }
    }
    
    _currTable.reverseMask();
    _currTable.intersectWithMask();
    _currTable.clearMask();

    
    
    int bitsPerWord=32;

    for (int i = 0; i < _supportSize; ++i){  
        
        //we initialize residues
        bool broken=false;
        for(int j=0; j<noTuples; j++){
            if(_supports[i*currTableSize+(j/bitsPerWord)]!=0x00000000 && !broken){
                _residues[i]=trail<int>(vars[0]->getSolver()->getStateManager(), j); 
                broken=true;
            }else{
                _residues[i]=trail<int>(vars[0]->getSolver()->getStateManager(), 0); 
            }
        }
    }
    
    //forall vars
    for (int i = 0; i < noVars; i++){
        if(_vars[i]->size()==1){
            if(tuplesOfSingletons[i]==-1){
                //printf("%%%%%% EMPTY DOMAIN 2\n");
                failNow();
                return;
            }
        }
    }
    if(_currTable.isEmpty()){
        //printf("%%%%%% EMPTY DOMAIN 3\n");
        failNow();
        return;
    }

}

void Table::post()
{
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

            //printf("%%%%%% var %d contains %d ",index, j);
            if(_vars[index]->contains(j)){
                //printf("YES \n");
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

void Table::filterDomains(){


    for(int i=0; i < _s_sup.size(); ++i){
        int index=_s_sup[i];
        //printf("%%%%%% filtering domain for var %d\n",index);
        for (int j = _vars[index]->min(); j <= _vars[index]->max(); j++){
            if(_vars[index]->contains(j)){ //i.e. a \in dom(x)
                int index_x_a=_supportOffsetJmp[index]+j-_vars[index]->initialMin();
                int indexResidue=_residues[index_x_a].value();

                if((_currTable._words[indexResidue] & _supports[(index_x_a)*currTableSize+indexResidue] ) == 0x00000000){
                    indexResidue=intersectIndexSparse(&_supports[index_x_a*currTableSize],_currTable);
                    
                    if(indexResidue!=-1){
                        _residues[index_x_a]=indexResidue; 
                    }else{
                        //printf("%%%%%% removing %d from %d\n",j+_vars[index]->initialMin(),index);
                        _vars[index]->remove(j);        
                    }
                  
                }
                
            }
        }
    }
}

void Table::enfoceGAC(){
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
void Table::printBits(unsigned int num) {
    // Extracting each bit of the int and printing it
    //yes rather weird function, but since we need to print %%%%%
    char str[32] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
    for (int i = 31; i >= 0; i--) {
        str[i] = (num >> i) & 1; 
        printf("%d",str[i]);
    }

    printf(" \n");
}