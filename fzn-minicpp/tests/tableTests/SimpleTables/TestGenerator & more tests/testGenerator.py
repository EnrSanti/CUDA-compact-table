#!/usr/bin/python3

import os 
import time
import random
import subprocess
from random import choice
from datetime import datetime
import gc


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



	#we generate also some additional constraints
	noConstraints=(int)(random.randint(1,12)*len(varsInTable)/100)
	constraints=""
	for i in range(noConstraints):
		constraintType=(int)(random.randint(1,12))
		valueC=random.randint(domainsMin[varsInTable[i]],domainsMax[varsInTable[i]])
		if(constraintType<2): #0,1 -> <
			constraints+="constraint x"+ str(varsInTable[i])+"<"+str(valueC)+";\n"
		elif(constraintType<3): #2 -> >
			constraints+="constraint x"+ str(varsInTable[i])+">"+str(valueC)+";\n" 
		elif(constraintType<8): #3,4,5,6,7 -> !=
			constraints+="constraint not(x"+ str(varsInTable[i])+"="+str(valueC)+");\n" 
		elif(constraintType<10): #add all different constraint
			varsDifferentStr=""	
			for v in varsInTable:
				#get 1 with 0.2 probability and 0 with 0.8
				if(random.randint(0,4)==1):
					varsDifferentStr+="x"+str(v)+","
			varsDifferentStr=varsDifferentStr[:-1]
			constraints+="constraint all_different(["+varsDifferentStr+"]);\n"
		 
			'''elif (constraintType<11):  # cumulative
			#noVarsIn= random.randint(int(len(varsInTable)/50),int(len(varsInTable)/30))
			noVarsIn= random.randint(0,int(len(varsInTable)))
			startTimes=""
			durations=""
			varsChosen=""
			availableResources=0
			#switch = random.randint(2,int(noVarsIn/10))
			switch = random.randint(2,int(noVarsIn))
			counter=0
			for i in range(noVarsIn):
				noVar=random.randint(1,len(varsInTable)-1)
				startTimes+=str(10*counter)+","
				durations+=str(20)+","
				varsChosen+="x"+str(varsInTable[noVar])+","
				availableResources+=int((domainsMax[varsInTable[noVar]]-domainsMin[varsInTable[noVar]])/2)
				if(i%switch==0):
					counter+=1
			
			availableResources=str(availableResources)
			durations=durations[:-1]
			startTimes=startTimes[:-1]
			varsChosen=varsChosen[:-1]
			#generate a random number of vars to be among	
			constraints+="constraint cumulative(["+startTimes+"],["+durations+"],["+varsChosen+"],"+availableResources+");\n"
			'''
		else: #countleq
			varToCount=""
			noVarsToLookFor=random.randint(int(len(varsInTable)/4),int(len(varsInTable)/3))
			for i in range(noVarsToLookFor):
				varToCount+="x"+str(varsInTable[random.randint(0,len(varsInTable)-1)])+","
			varToCount=varToCount[:-1]
			varToCountLast="x"+str(varsInTable[-1])
			constraints="constraint count_leq(["+varToCount+"],1000,"+varToCountLast+");\n"
	return table,constraints

def remove_up_to_first_newline(input_string):
    # Find the position of the first newline and slice the string
    return input_string.split('\n', 1)[1] if '\n' in input_string else input_string


def generateFile():
	global include
	fileStr=""
	fileStrCUDA=""
	fileTmp=""
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
	fileTmp=fileStr
	
	#generate between 2 and 6 tables
	noTables=random.randint(2, 4)
	for tblNo in range(0,noTables):
		noVarsInTable=random.randint((int)(noVars*0.2),(int)(noVars*0.4))
		varsInTable=[]
		for i in range(noVarsInTable):
			varsInTable.append(generate_random(0,noVars,varsInTable))

		print(varsInTable)
		table,otherConstraint=generateConstraints(varsInTable,domainsMin,domainsMax,noTuples,tblNo)

		constraintLine="constraint table(["
		for xNo in varsInTable:
			constraintLine+="x"+str(xNo)+","
		constraintLine=constraintLine[:-1]+"],t"+str(tblNo)

		constraintLine+=")"

		#yes we duplicate, it's not optimal
		fileStrCUDA+=table+constraintLine+"::gpu;\n"
		fileTmp+=table+constraintLine+";\n"
		fileStr+=table+constraintLine+";\n"
		#add the other constraints
		fileStrCUDA+=otherConstraint
		fileTmp+=otherConstraint
		fileStr+=otherConstraint

	fileStrCUDA+="solve minimize x1;"
	fileStr+="solve minimize x1;"
	fileTmp+=otherConstraint+"solve minimize x1;"
	return fileStr, fileStrCUDA, fileTmp
############################  MODIFIABLE VARIABLES  ################################
####################################################################################

#note, it doesn't create n sat instances and n unsat instances, but it create n instances, then they are solved via cp_model and put in the right (SAT or NOT folder)
filesToCreate=40

#how many clauses we want in an instance (max and min)
minNoVars=10
maxNoVars=30

minDomain=20
maxDomain=80
maxOffset=3


minTuples=60
maxTuples=200

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
random.seed(402)



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
	directoryPathUNSAT="testsUNSAT_shallow3/"
	directoryPathSAT="testsSAT_shallow3/"
	directoryPathUNSAT_CUDA="testsUNSAT_CUDA_shallow3/"
	directoryPathSAT_CUDA="testsSAT_CUDA_shallow3/"

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
	fileStr,fileCUDA,fileTmp = generateFile()

	#save file tmp
	with open('tmp.mzn', 'w') as f:
		f.write(remove_up_to_first_newline(remove_up_to_first_newline(remove_up_to_first_newline(fileTmp))))
		f.close()
	
	t0=time.time()
	print("Solving model (t0="+str(datetime.fromtimestamp(t0))+")...")
	result = subprocess.run(["minizinc","--solver","Geeecode","./tmp.mzn"], capture_output=True, text=True)
	print("model solved in "+str(time.time()-t0)+" seconds")	
	status=""
	print(result.stdout)
	if("=====UNSATISFIABLE=====" in result.stdout):
		status="UNSAT"
	elif(not ("=====ERROR=====") in result.stdout):
		status="SAT"
	else:
		status="ERROR"
		print(f"\033[91m FAILED\033[00m"+result.stdout+result.stderr)
	t1=time.time()



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
		f_CUDA.write(fileCUDA)

	del fileStr
	del fileCUDA
	del fileTmp
	gc.collect()
