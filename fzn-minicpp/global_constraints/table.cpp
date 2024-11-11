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
        //calculating the number of rows in the support bitset
        _supportSize+=vars[i]->intialSize();
        //we store the offset
        _variablesOffsets[i]=vars[i]->min();      
        printf("%%%%%% offsets: %d\n",_variablesOffsets[i]);

    }
    printf("%%%%%% supportSize (sum of the vars dom): %d\n",_supportSize);


    //calculating the offset of the variables, used in accessing the support rows    
    _supportOffsetJmp[0]=0;
    for (int i = 1; i < noVars; i++){
        _supportOffsetJmp[i]=_supportOffsetJmp[i-1]+vars[i-1]->intialSize();
        //print 
        printf("%%%%%% supportOffsetJmp[%d]: %d\n",i,_supportOffsetJmp[i]);
    }

    //we allocate and initialize the support bitsets
    printf("%%%%%% creating _supports (%d of them) of size %d\n",_supportSize,noTuples);
    _supports=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _residues=vector<trail<int>>(_supportSize);


    
    
    bool found=false;
    //-1 means that no value in the domain of the variable is in any table row
    int tuplesOfSingletons[noVars];

    for (int v = 0; v < noVars; v++){
        tuplesOfSingletons[v]=-1;
        for (int t = 0; t < noTuples; t++){
            if(tuples[t][v]>=_vars[v]->min() && tuples[t][v]<=_vars[v]->max()){
                //classical entry (we update all the supports in the same way)
                int entryValue=tuples[t][v]-_variablesOffsets[v];   
                
                int offset=_supportOffsetJmp[v]+entryValue;

                _supports[offset].addToMaskInt(t+1); 

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

    //qui non è e,mpty
    
    int bitsPerWord=32;

    for (int i = 0; i < _supportSize; ++i){  
        _supports[i].intersectWithMask();

        _supports[i].clearMask();
        
        //we initialize residues
        bool broken=false;
        for(int j=0; j<noTuples; j++){
            if(_supports[i]._words[j/bitsPerWord].value()!=0x00000000 && !broken){
                _residues[i]=trail<int>(vars[0]->getSolver()->getStateManager(), j); 
                broken=true;
            }else{
                _residues[i]=trail<int>(vars[0]->getSolver()->getStateManager(), 0); 
            }
        }
    }

    printf("%%%%%% FIN QUI C'ARRIVO\n");

    //_currTable.print(0);
    for (int i = 0; i < noVars; i++){
        if(_vars[i]->size()==1){
            //tuplesOfSingletons[v]=-1 SSE NESSUN VALORE nel dominio per la var v è nella tbl
            printf("%%%%%% tuplesOfSingletons[i]: %d\n",tuplesOfSingletons[i]);
            if(tuplesOfSingletons[i]==-1){
                failNow();
                return;
            }
        }
    }

    printf("%%%%%% FIN QUI C'ARRIVO 2\n");
    if(_currTable.isEmpty()){
        failNow();
        return;
    }
    _currTable.print(0);
}

void Table::post()
{
    for (auto const & v : _vars){
       v->propagateOnBoundChange(this);
    }
}

void Table::propagate()
{
    printf("%%%%%% ******** propagating: ********\n");
    printf("%%%%%% _currTable: \n");
    //_currTable.print(0);
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
        vector<int> dom=_vars[index]->dumpDomainToVec();
        for (int j = 0; j < dom.size(); j++){ 
            int index_x_a=_supportOffsetJmp[index]+dom[j]-_variablesOffsets[index];
            _currTable.addToMaskVector(_supports[index_x_a]._words);
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
        for (int j = 0; j < _vars[index]->size(); j++){
            if(_vars[index]->contains(j+_vars[index]->initialMin())){ //i.e. a \in dom(x)

                int index_x_a=_supportOffsetJmp[index]+j;
                int indexResidue=_residues[index_x_a].value();

                if((_currTable._words[indexResidue] & _supports[index_x_a]._words[indexResidue] ) == 0x00000000){
                
                    indexResidue=_supports[index_x_a].intersectIndexSparse(_currTable);
                    
                    if(indexResidue!=-1){
                        _residues[index_x_a].setValue(indexResidue); //ok setVal
                    }else{
                        _vars[index]->remove(j+_vars[index]->initialMin());                   
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
    
	for (int i = 0; i < _vars.size(); i++){
		//update s_val and the deltas
        if(_vars[i]->changed()){
            printf("%%%%%% var %d changed\n",i);
            //for each val of the var print it
            vector<int> dom=_vars[i]->dumpDomainToVec();
            /*
            for (int j = 0; j < dom.size(); j++){
                if(dom[j]!=0)
                    printf("%%%%%% var[%d] contains %d\n",i,dom[j]);
            }*/
            _s_val.push_back(i);
        }
        
		//update s_sup
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
        }
	}
	updateTable();	
	filterDomains();
}