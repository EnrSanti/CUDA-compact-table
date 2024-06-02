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
    _deltaXs=vector<SparseBitSet>(noVars,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),0));
    _lastVarsValues=vector<SparseBitSet>(noVars,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),0));

    for (int i = 0; i < noVars; i++){        
        //calculating the number of rows in the support bitset
        _supportSize+=vars[i]->intialSize();
        //we store the offset
        _variablesOffsets[i]=vars[i]->min();
        //we allocate the delta and lastVarsValues
        _deltaXs[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),_vars[i]->size());
        _lastVarsValues[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),_vars[i]->size());

        //initialize lastVarsValues
        vars[i]->dumpInSparseBitSet(vars[i]->min(),vars[i]->max(),_lastVarsValues[i]);

    }

    //calculating the offset of the variables, used in accessing the support rows    
    _supportOffsetJmp[0]=0;
    for (int i = 1; i < noVars; i++){
        _supportOffsetJmp[i]=_supportOffsetJmp[i-1]+vars[i-1]->size();
    }

    //we allocate and initialize the support bitsets
    _supports=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _supportsShort=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _supportsMin=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _supportsMax=vector<SparseBitSet>(_supportSize,SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples));
    _residues= vector<trail<int>>(_supportSize);


    //we allocate and initialize the support bitsets
    for (int i = 0; i < _supportSize; i++){
        _supports[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);//the content doesn't make sense yet, later we need to update the mask and intersect it
        _supportsShort[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        _supportsMax[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
        _supportsMin[i]=SparseBitSet(vars[0]->getSolver()->getStateManager(),vars[0]->getSolver()->getStore(),noTuples);
    }



    for (int t = 0; t < noTuples; t++){
        for (int v = 0; v < noVars; v++){
            
            if(tuples[t][v]>=_vars[v]->min() && tuples[t][v]<=_vars[v]->max()){
                //classical entry (we update all the supports in the same way)
                int entryValue=tuples[t][v]-_variablesOffsets[v];   
                
                int offset=_supportOffsetJmp[v]+entryValue;

                //TO UNCOMMENT ONCE ALL ENTRIES ARE IMLPMENTED
                //we check if it's not a short entry (i.e. has *)
                /*if(strncmp(var,"*",1)==0){
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
                }else{*/
            
                _supports[offset].addToMaskInt(t+1); 
                _supportsShort[offset].addToMaskInt(t+1);

                for (int varValue = 0; varValue <= entryValue; varValue++) {
                    offset=_supportOffsetJmp[v]+varValue;
                    //update supportsMin
                    _supportsMin[offset].addToMaskInt(t+1);  
                }
                for (int varValue = entryValue; varValue < vars[v]->intialSize(); varValue++) {
                    offset=_supportOffsetJmp[v]+varValue;
                    //update supportsMax
                    _supportsMax[offset].addToMaskInt(t+1);
                }
            }
        }
    }
    

    int bitsPerWord=32;

    for (int i = 0; i < _supportSize; ++i){  
        _supports[i].intersectWithMask();
        _supportsShort[i].intersectWithMask();
        _supportsMin[i].intersectWithMask();
        _supportsMax[i].intersectWithMask();

        _supports[i].clearMask();
        _supportsShort[i].clearMask();
        _supportsMin[i].clearMask();
        _supportsMax[i].clearMask();
        
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
    //print();
    
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
        if(_deltaXs[index].countOnes() < _vars[index]->size()){
            //incremental update
            printf("%%%%%% incremental update \n");
            
            for (int j = 0; j < _vars[index]->intialSize(); j++){
                if(_deltaXs[index].getIthBit(j)==1){                    
                    int index_x_a=_supportOffsetJmp[index]+j;
                    printf("%%%%%% incremental update \n");
                    _currTable.addToMaskVector(_supports[index_x_a]._words);
                }
            }    
			
            _currTable.reverseMask();
				
            // TODO UNCOMMENT
            //if(dom(i).minChanged()){
            //	...	
            //}
            //if(dom(i).maxChanged()){
            //	...
            //}
        }else{
            //reset based update
            printf("%%%%%% reset based update \n");
            vector<int> dom=_vars[index]->dumpDomainToVec();
            
            for (int j = 0; j < dom.size(); j++){ 
                int index_x_a=_supportOffsetJmp[index]+dom[j]-_variablesOffsets[index];
                _currTable.addToMaskVector(_supports[index_x_a]._words);
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
        //update lastVarValues (all words)
        _vars[index]->dumpInSparseBitSet(_vars[index]->min(),_vars[index]->max(),_lastVarsValues[index]);
    }
    
}

void Table::enfoceGAC(){
    //update the table
    _s_val.clear();
    _s_sup.clear();
	for (int i = 0; i < _vars.size(); i++){
		//update s_val and the deltas
        if(_vars[i]->changed()){
            _s_val.push_back(i);
            updateDelta(i);
        }
		//update s_sup
        if(_vars[i]->size()>1){
            _s_sup.push_back(i);
        }
	}

	updateTable();
	
	filterDomains();
}


void Table::updateDelta(int i){

    _vars[i]->dumpInSparseBitSet(_vars[i]->min(),_vars[i]->max(),_deltaXs[i]);
    //we calculate the delta by xoring the words
    for (int j = 0; j < _vars[i]->getSizeOfBitSet(); j++){
        _deltaXs[i]._words[j].setValue(_deltaXs[i]._words[j].value()^_lastVarsValues[i]._words[j].value()); //BEWARE, BROKEN THE DATA STRUCTURE can be replaced with x XOR y = (x AND (NOT y)) OR ((NOT x) AND y)
    }
}

void Table::print(){    
    printf("%%%%%% ----------------- TABLE PRINT: -----------------\n\n");

    printf("%%%%%% Size of currTable (no of rows): %ld \n", _tuples.size());
    printf("%%%%%% The table (support) has %ld vars\n",_vars.size());
    for (int i = 0; i < _vars.size(); i++){
        printf("%%%%%% Var size: %d, words no: %d \n",_vars[i]->size(),_vars[i]->getSizeOfBitSet());
    }
    int currentOffset=0;
    int internalOffset=0;
    int offsetAccumulator=_vars[0]->intialSize();
	printf("%%%%%% ------ var %d in the table has current size: %d ------\n",currentOffset,_vars[currentOffset]->size());
    
    for (int i = 0; i < _supportSize; i++){
        if(i>=offsetAccumulator){
    		currentOffset++;
    		offsetAccumulator+=_vars[currentOffset]->intialSize();
    		internalOffset=0;
    		printf("%%%%%% ------ var %d in the table has current size: %d ------\n",currentOffset,_vars[currentOffset]->size());
    	}
        _supports[i].print(_variablesOffsets[currentOffset]+internalOffset);
        internalOffset++;
    }
    printf("%%%%%% ----------------- VARS: -----------------\n\n");
    for (int i = 0; i < _vars.size(); i++){
        printf("%%%%%% Var %d: %d\n",i,_vars[i]->getId());

        //_vars[i]->dump(_vars[i]->min(),_vars[i]->max(),_deltaXs[i]);
        
        printf("%%%%%% Var %d dump: %d \n",i,_lastVarsValues[i]._words[0].value());
    }
    for (int i = 0; i < _vars.size(); i++){
        //checking the contained values
        for (int j = 0; j < _vars[i]->intialSize(); j++){
            if(_vars[i]->contains(j+_vars[i]->initialMin()))
                printf("%%%%%% Var %d contains? %d: YES\n",i,j+_vars[i]->initialMin());
            else
                printf("%%%%%% Var %d contains? %d: NO\n",i,j+_vars[i]->initialMin());
        }
    }
}