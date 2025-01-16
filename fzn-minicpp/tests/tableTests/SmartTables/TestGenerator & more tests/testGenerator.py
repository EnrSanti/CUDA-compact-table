#!/usr/bin/python3

import os 
import random
import time
from random import choice
from ortools.constraint_solver import pywrapcp	
from ortools.sat.python import cp_model
from datetime import datetime
import numpy as np
import gc

model=None

#generate a value in [from_,to_] if not in already in notThese
def generate_random(from_,to_, notThese):
    result = choice([i for i in range(from_,to_) if (i not in notThese)])
    return result

def init_model():
	global model
	model = cp_model.CpModel()

def del_model():
	global model
	del model
	gc.collect()

def generateConstraints(varsInTable,domainsMin,domainsMax,noTuples,tableNo):

	table="array [int,int] of "
	signTable="array [int,int] of SmartTableOp: signs"+str(tableNo) + "=[|"
	minV=min(domainsMin)
	maxV=max(domainsMax)
	table=table+str(minV)+".."+str(maxV)+" : "+"t"+str(tableNo)+"=[|"
	tuples=[]
	
	#ŧesting the satifsfiability
	varsInSolver=[]

	for v in varsInTable:
		varsInSolver.append(model.new_int_var(domainsMin[v],domainsMax[v], "x"+str(v)))

	#print the size of varsInSolver
	print("size of varsInSolver:"+str(len(varsInSolver)))
	disjuncts=[]
	for i in range(noTuples):
		#generate tuple
		expr=None
		literals=[]

		constrTuple=model.NewBoolVar("constrTuple"+str(i))
		for v in varsInTable:
			#print v

			sign=np.random.choice(["Int","GtInt","LtInt","All"], p=[0.6, 0.15, 0.15,0.1])
			val=random.randint(domainsMin[v],domainsMax[v])
	

			if(sign=="Int"):
				lit=model.NewBoolVar("lit"+str(i)+str(v))

				model.add(varsInSolver[varsInTable.index(v)]==val).OnlyEnforceIf(lit)

				literals.append(lit)
				
			elif(sign=="GtInt"):
				lit=model.NewBoolVar("lit"+str(i)+str(v))

				model.add(varsInSolver[varsInTable.index(v)]>val).OnlyEnforceIf(lit)

				literals.append(lit)

			elif(sign=="LtInt"):
				lit=model.NewBoolVar("lit"+str(i)+str(v))

				model.add(varsInSolver[varsInTable.index(v)]<val).OnlyEnforceIf(lit)

				literals.append(lit)
			elif(sign=="All"):
				#model.add(varsInSolver[v]<=domainsMax[v]).OnlyEnforceIf(lit) #always sat
				pass
			else:
				print("ERROR")
				exit(1)

			#add value to the table to put in the file
			signTable+=str(sign)+","
			table+=str(val)+","


		model.AddBoolAnd(literals).OnlyEnforceIf(constrTuple)
		disjuncts.append(constrTuple)

		#not efficient removal of "," form the last el
		table = table[:-1]
		signTable = signTable[:-1]
		table+="|\n"
		signTable+="|"
	
	model.AddBoolOr(disjuncts)

	table = table[:-1]
	signTable+="]; \n"
	table+="]; \n"


	#we generate also some additional constraints
	noConstraints=(int)(random.randint(1,12)*len(varsInTable)/100)
	constraints=""
	for i in range(noConstraints):
		constraintType=(int)(random.randint(1,12))
		valueC=random.randint(domainsMin[varsInTable[i]],domainsMax[varsInTable[i]])
		if(constraintType<2): #0,1 -> <
			constraints+="constraint x"+ str(varsInTable[i])+"<"+str(valueC)+";\n"
			model.add(varsInSolver[i]<valueC)
		elif(constraintType<3): #2 -> >
			constraints+="constraint x"+ str(varsInTable[i])+">"+str(valueC)+";\n" 
			model.add(varsInSolver[i]>valueC)
		elif(constraintType<8): #3,4,5,6,7 -> !=
			constraints+="constraint not(x"+ str(varsInTable[i])+"="+str(valueC)+");\n" 
			model.add(varsInSolver[i]!=valueC)
		elif(constraintType<10): #add all different constraint
			varsDifferentStr=""	
			varsDiffModel=[]
			for v in varsInTable:
				#get 1 with 0.2 probability and 0 with 0.8
				if(random.randint(0,4)==1):
					varsDifferentStr+="x"+str(v)+","
					varsDiffModel.append(varsInSolver[varsInTable.index(v)])	
			varsDifferentStr=varsDifferentStr[:-1]
			constraints+="constraint all_different(["+varsDifferentStr+"]);\n"
			model.AddAllDifferent(varsDiffModel)
		else: #countleq
			varToCount=""
			noVarsToLookFor=random.randint(int(len(varsInTable)/4),int(len(varsInTable)/3))
			for i in range(noVarsToLookFor):
				varToCount+="x"+str(varsInTable[random.randint(0,len(varsInTable)-1)])+","
			varToCount=varToCount[:-1]
			varToCountLast="x"+str(varsInTable[-1])
			constraints="constraint count_leq(["+varToCount+"],1000,"+varToCountLast+");\n"
			model.add(cp_model.LinearExpr.Sum([varsInSolver[varsInTable.index(v)] for v in varsInTable])<=1000)
	return signTable+table, constraints



def remove_up_to_first_newline(input_string):
    # Find the position of the first newline and slice the string
    return input_string.split('\n', 1)[1] if '\n' in input_string else input_string


def generateFile():
	global model
	global include
	fileStr=""
	fileStrCUDA=""
	
	noTuples=random.randint(minTuples, maxTuples)
	noVars=random.randint(minNoVars, maxNoVars)
	print("noTuples:"+str(noTuples))
	print("noVars:"+str(noVars))
	fileStr=include
	
	domainsMin=[]
	domainsMax=[]

	for i in range(noVars):
		size=random.randint(minDomain, maxDomain)
		offset=random.randint(1,maxOffset)
		domainsMin.append(offset)
		domainsMax.append(offset+size)
		toWrite="var "+str(domainsMin[i])+".."+str(domainsMax[i])+" : "+"x"+str(i)+";\n"
		fileStr+=toWrite

	fileStrCUDA=fileStr
	#generate between 2 and 6 tables
	noTables=random.randint(2, 3)
	init_model()
	for tblNo in range(0,noTables):
		noVarsInTable=random.randint((int)(noVars*0.2),(int)(noVars*0.4))
		varsInTable=[]
		for i in range(noVarsInTable):
			varsInTable.append(generate_random(0,noVars,varsInTable))

		print(varsInTable)
		table,otherConstraint=generateConstraints(varsInTable,domainsMin,domainsMax,noTuples,tblNo)

		constraintLine="constraint smart_table(["
		for xNo in varsInTable:
			constraintLine+="x"+str(xNo)+","
		constraintLine=constraintLine[:-1]+"],t"+str(tblNo)+",signs"+str(tblNo)

		constraintLine+=")"

		#yes we duplicate, it's not optimal
		fileStrCUDA+=table+constraintLine+"::gpu;\n"
		fileStr+=table+constraintLine+";\n"
		#add the other constraints
		fileStrCUDA+=otherConstraint
		fileStr+=otherConstraint
	

	solver = cp_model.CpSolver()
	t0=time.time()
	print("Solving model (t0="+str(datetime.fromtimestamp(t0))+")...")
	
	status = solver.solve(model)
	
	t1=time.time()

	print("model solved in "+str(time.time()-t0)+" seconds")	
	
	if(status==cp_model.OPTIMAL or status==cp_model.FEASIBLE):
		print("SAT")
		retStat="SAT"
	elif(status==cp_model.INFEASIBLE):
		print("UNSAT")
		retStat="UNSAT"
	else:
		retStat="TIMEOUT"
		print("...TIMEOUT\n")

	
	del_model()
	fileStrCUDA+="solve satisfy;"
	fileStr+="solve satisfy;"
	return retStat,fileStr, fileStrCUDA


############################  MODIFIABLE VARIABLES  ################################
####################################################################################

#note, it doesn't create n sat instances and n unsat instances, but it create n instances, then they are solved via cp_model and put in the right (SAT or NOT folder)
filesToCreate=40

#how many clauses we want in an instance (max and min)
minNoVars=20
maxNoVars=50

minDomain=8
maxDomain=45
maxOffset=3


minTuples=10
maxTuples=80

osType="linux"; # "windows" or "linux" #used just to specify the directory format

#cosntant string reported before each file
include="""%test automatically generated\n
include \"minicpp.mzn\";
include \"table.mzn\";\n
include \"cumulative.mzn\";\n
include \"all_different.mzn\"; \n
include \"count_leq.mzn\"; \n\n"""

####################################################################################
####################################################################################


#we set the seed so we always generate that instances
random.seed(424)
np.random.seed(424)


directoryPathUNSAT=""
directoryPathSAT=""
directoryPathUNSAT_CUDA=""
directoryPathSAT_CUDA=""

#select the proper path
if(osType=="windows"):
	directoryPathUNSAT="testsUNSAT\\"
	directoryPathSAT="testsSAT\\"
	directoryPathUNSAT_CUDA="testsUNSAT_CUDA\\"
	directoryPathSAT_CUDA="testsSAT_CUDA\\"
else:
	directoryPathUNSAT="testsUNSAT_big2/"
	directoryPathSAT="testsSAT_big2/"
	directoryPathUNSAT_CUDA="testsUNSAT_CUDA_big2/"
	directoryPathSAT_CUDA="testsSAT_CUDA_big2/"

#check if folders exist else create them
if not os.path.isdir(directoryPathSAT):
	os.makedirs(directoryPathSAT) 
if not os.path.isdir(directoryPathUNSAT):
	os.makedirs(directoryPathUNSAT) 
if not os.path.isdir(directoryPathSAT_CUDA):
	os.makedirs(directoryPathSAT_CUDA) 
if not os.path.isdir(directoryPathUNSAT_CUDA):
	os.makedirs(directoryPathUNSAT_CUDA) 

satFilesNo=0
unsatFilesNo=0
#we generate the different .mzn files
for i in range(1,filesToCreate+1):
	#the size of n
	where,fileStr,fileCUDA = generateFile()


	folder=""
	folder_cuda=""
	counter=0
	if(where=="SAT"):
		print("put in SAT FOLDER")
		folder=directoryPathSAT
		folder_cuda=directoryPathSAT_CUDA
		satFilesNo+=1
		counter=satFilesNo
	else:
		print("put in UNSAT FOLDER")
		folder=directoryPathUNSAT
		folder_cuda=directoryPathUNSAT_CUDA
		unsatFilesNo+=1
		counter=unsatFilesNo



	with open(folder+'test_'+str(counter)+'.mzn', 'w') as f:
		f.write(fileStr)
	with open(folder_cuda+'test_'+str(counter)+'.mzn', 'w') as f_CUDA:
		f_CUDA.write(fileCUDA)

	del fileStr
	del fileCUDA
	gc.collect()
