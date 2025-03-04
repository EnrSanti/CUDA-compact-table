#!/usr/bin/python3

import os 
import time
import random
import subprocess
from random import choice
from datetime import datetime
import gc

import itertools
#WARNING: THIS PROGRAM ISN't OPTIMIZED 
#could be done much better, anyway, it's just to create instances
#still now with maps is somehow usable


#generate a value in [from_,to_] if not in already in notThese
#the following function is taken from: https://www.w3resource.com/python-exercises/list/python-data-type-list-exercise-145.php
def generate_random(from_,to_, notThese):
    result = choice([i for i in range(from_,to_) if (i not in notThese)])
    return result

def generateConstraints(varsInTable,domainsMin,domainsMax,noTuples,tableNo):
    num_vars = len(varsInTable)

    # Initialize equation components
    constraints = "constraint "
    absMin, absMax = 0, 0
    coefficients = []

    # Generate coefficients and build the equation
    for i in range(num_vars):
        valueC = 2  # Fixed coefficient
        coefficients.append(valueC)
        constraints += f"{valueC}*x{varsInTable[i]}+"
        absMin += valueC * domainsMin[varsInTable[i]]
        absMax += valueC * domainsMax[varsInTable[i]]
        print(absMax)
    # Choose a value for the equation
    valueC = random.randint(absMin, absMax-1)
    if(valueC%2==1):
        valueC+=1
    constraints = constraints[:-1]  # Remove the trailing "+"
    constraints += f"={valueC};\n"

    # Generate ordered tuples
    domain_ranges = [range(domainsMin[v], domainsMax[v] + 1) for v in varsInTable]
    tuples = []

    for t in itertools.islice(itertools.product(*domain_ranges), noTuples - 1):
        tuples.append(t)


    # **Find or create a tuple that satisfies the equation**
    satisfying_tuple = None
    

    
# **If no valid tuple was found, create one by adjusting multiple variables**
    if satisfying_tuple is None:
        # Start with minimum values for all variables
        base_tuple = [domainsMin[v] for v in varsInTable]

        # Compute the initial sum with all min values
        current_sum = sum(coefficients[i] * base_tuple[i] for i in range(num_vars))

        # Distribute the difference across multiple variables
        difference = valueC - current_sum
        i = 0
        while difference != 0 and i < num_vars:
            max_possible_increase = domainsMax[varsInTable[i]] - base_tuple[i]
            change = min(difference // coefficients[i], max_possible_increase)
            base_tuple[i] += change
            difference -= change * coefficients[i]
            i += 1  # Move to the next variable if adjustment is needed

        satisfying_tuple = tuple(base_tuple)

    tuples.append(satisfying_tuple)

    # Construct the output table
    table = f"array [int,int] of {min(domainsMin)}..{max(domainsMax)} : t{tableNo}=[|\n"
    for t in tuples:
        table += ",".join(map(str, t)) + "|\n"
    table = table.rstrip("\n") + "]; \n"

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
		offset=1
		domainsMin.append(offset)
		domainsMax.append(offset+size)
		toWrite="var "+str(domainsMin[i])+".."+str(domainsMax[i])+" : "+"x"+str(i)+";\n"
		fileStr+=toWrite

	fileStrCUDA=fileStr
	fileTmp=fileStr
	
	#generate between 2 and 6 tables
	noTables=1
	for tblNo in range(0,noTables):
		noVarsInTable=random.randint((int)(noVars*0.6),(int)(noVars*0.8))
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

	fileStrCUDA+="solve satisfy;"
	fileStr+="solve  satisfy;"
	fileTmp+="solve  satisfy;"
	return fileStr, fileStrCUDA, fileTmp

############################  MODIFIABLE VARIABLES  ################################
####################################################################################

#note, it doesn't create n sat instances and n unsat instances, but it create n instances, then they are solved via cp_model and put in the right (SAT or NOT folder)
filesToCreate=40

#how many clauses we want in an instance (max and min)
minNoVars=30
maxNoVars=30

minDomain=110
maxDomain=110
maxOffset=0


minTuples=999
maxTuples=4999


#cosntant string reported before each file
include="""%test automatically generated\n
include \"minicpp.mzn\";
include \"table.mzn\";\n\n"""
####################################################################################
####################################################################################


#we set the seed so we always generate that instances
random.seed(4050)


directoryPathUNSAT="oneSolUNSAT_30_110_5000/"
directoryPathSAT="oneSolSAT_30_110_5000/"
directoryPathUNSAT_CUDA="oneSolUNSAT_CUDA_30_110_5000/"
directoryPathSAT_CUDA="oneSolSAT_CUDA_30_110_5000/"

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
	result = subprocess.run(["minizinc","--solver","Gecode","./tmp.mzn"], capture_output=True, text=True)
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
		print("writing to"+str(folder_cuda+'test_'+str(counter)+'.mzn'))
		f_CUDA.write(fileCUDA)

	#print(fileCUDA)
	del fileStr
	del fileCUDA
	del fileTmp
	gc.collect()
