#!/usr/bin/python3

import os 
import random
from random import choice
from ortools.sat.python import cp_model

#WARNING: THIS PROGRAM ISN't OPTIMIZED 
#could be done much better, anyway, it's just to create instances
#still now with maps is somehow usable


#generate a value in [from_,to_] if not in already in notThese
#the following function is taken from: https://www.w3resource.com/python-exercises/list/python-data-type-list-exercise-145.php
def generate_random(from_,to_, notThese):
    result = choice([i for i in range(from_,to_) if (i not in notThese)])
    return result

def generateConstraints(varsInTable,domainsMin,domainsMax,noTuples,tableNo):

	table="array [int,int] of "
	minV=min(domainsMin)
	maxV=max(domainsMax)
	table=table+str(minV)+".."+str(maxV)+" : "+"t"+str(tableNo)+"=[|"
	tuples=[]
	for i in range(noTuples):
		#generate tuple
		t=[]
		for v in varsInTable:
			val=random.randint(domainsMin[v],domainsMax[v])
			t.append(val)
			#add value to the table to put in the file
			table+=str(val)+","
		#not efficient removal of "," form the last el
		table = table[:-1]
		table+="|\n"
		tuples.append(tuple(t))
	table = table[:-1]
	table+="]; \n"


	#ŧesting the satifsfiability
	model = cp_model.CpModel()
	varsInSolver=[]

	for v in varsInTable:
		varsInSolver.append(model.new_int_var(domainsMin[v],domainsMax[v], "x"+str(v)))

	for i in range(noTuples):
		model.AddAllowedAssignments(varsInSolver,tuples)

	#we generate also some additional constraints
	noConstraints=(int)(random.randint(0,8)/10*len(varsInTable))
	constraints=""
	for i in range(noConstraints):
		#choosing the constraint > < != or =
		constraintType=(int)(random.randint(1,10))
		valueC=random.randint(domainsMin[varsInTable[i]],domainsMax[varsInTable[i]])
		if(constraintType<2): #0,1 -> <
			model.add(varsInSolver[i]<valueC)
			constraints+="constraint x"+ str(varsInTable[i])+"<"+str(valueC)+";\n"
		elif(constraintType<4): #2,3 -> >
			model.add(varsInSolver[i]>valueC)
			constraints+="constraint x"+ str(varsInTable[i])+">"+str(valueC)+";\n" 
		
		elif(constraintType<10): #4,5,6,7,8,9 -> !=
			model.add(varsInSolver[i]!=valueC)
			constraints+="constraint not(x"+ str(varsInTable[i])+"="+str(valueC)+");\n" 
		
		else: #10 -> =
			model.add(varsInSolver[i]==valueC)
			constraints+="constraint x"+ str(varsInTable[i])+"="+str(valueC)+";\n"
		
	solver = cp_model.CpSolver()


	status = solver.solve(model)
	retStat=""
	if(status==cp_model.OPTIMAL or status==cp_model.FEASIBLE):
		print("SAT")
		retStat="SAT"
	elif(status==cp_model.INFEASIBLE):
		print("UNSAT")
		retStat="UNSAT"
	else:
		print("...SOMETHING'S WRONG\n")
	
	return retStat,table,constraints

############################  MODIFIABLE VARIABLES  ################################
####################################################################################

#note, it doesn't create n sat instances and n unsat instances, but it create n instances, then they are solved via cp_model and put in the right (SAT or NOT folder)
filesToCreate=40

#how many clauses we want in an instance (max and min)
minNoVars=4
maxNoVars=10

minDomain=20
maxDomain=400
maxOffset=200

#minNoTables=1 #not yet used only 1 table
#maxNoTables=1

minTuples=5
maxTuples=200

osType="linux"; # "windows" or "linux" #used just to specify the directory format

#cosntant string reported before each file
include="""%test automatically generated\n
include \"table.mzn\";
include \"minicpp.mzn\";\n\n"""

####################################################################################
####################################################################################


#we set the seed so we always generate that instances
random.seed(500)



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
	directoryPathUNSAT="testsUNSAT2/"
	directoryPathSAT="testsSAT2/"
	directoryPathUNSAT_CUDA="testsUNSAT_CUDA2/"
	directoryPathSAT_CUDA="testsSAT_CUDA2/"

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
	fileStr=""
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

	noVarsInTable=random.randint((int)(noVars*0.3),(int)(noVars*0.7))
	varsInTable=[]
	for i in range(noVarsInTable):
		varsInTable.append(generate_random(0,noVars,varsInTable))

	print(varsInTable)
	status,table,otherConstraint=generateConstraints(varsInTable,domainsMin,domainsMax,noTuples,0)
	fileStr+=table

	constraintLine="constraint table(["
	for xNo in varsInTable:
		constraintLine+="x"+str(xNo)+","
	constraintLine=constraintLine[:-1]+"],t"+str(0)

	constraintLine+=")"

	#yes we duplicate, it's not optimal
	fileStrCUDA=fileStr+constraintLine+"::gpu;\n"
	fileStr+=constraintLine+"::uniud;\n"

	#add the other constraints
	fileStrCUDA+=otherConstraint
	fileStr+=otherConstraint
	fileStrCUDA+="solve satisfy;"
	fileStr+="solve satisfy;"


	folder=""
	folder_cuda=""
	counter=0
	if(status=="SAT"):
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
		f_CUDA.write(fileStrCUDA)

