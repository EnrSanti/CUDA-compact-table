import os
import subprocess
import time

#it just check sat/unsat (on the small instances you can also check the assignments)

# List of model files
modelsSAT = ["serial/SAT/",
"CUDA/SAT/",
"TestGenerator & more tests/testsSAT/",
"TestGenerator & more tests/testsSAT_CUDA/",
"TestGenerator & more tests/testsSAT_bigger/",
"TestGenerator & more tests/testsSAT_CUDA_bigger/",
"TestGenerator & more tests/testsSAT_even_bigger/",
"TestGenerator & more tests/testsSAT_CUDA_even_bigger/"]

modelsUNSAT = ["serial/UNSAT/",
"CUDA/UNSAT/",
"TestGenerator & more tests/testsUNSAT/",
"TestGenerator & more tests/testsUNSAT_CUDA/",
"TestGenerator & more tests/testsUNSAT_bigger/",
"TestGenerator & more tests/testsUNSAT_CUDA_bigger/",
"TestGenerator & more tests/testsUNSAT_even_bigger/",
"TestGenerator & more tests/testsUNSAT_CUDA_even_bigger/"]

solver="MiniCpp"
# Run each serial SAT model
print("Running simple models")

totalErrors=0

def run_sat_models(prefix):
    global totalErrors
    for folder in modelsSAT:  
        folder=prefix+folder
        print("Running SAT models in "+folder+" -> ",end="")

        if not os.path.exists(folder):
            print(f"\033[93m FOLDER not found, SKIPPING\033[00m")
            continue
        instances = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
        #forall files in the folder get their name
        #print all instances
        errors=0
        t0 = time.time()
        #check if folder exists, print in yellow
        
        for instance in instances:
            result = subprocess.run(["minizinc", "--solver", solver,folder+instance], capture_output=True, text=True)
            if("=====UNSATISFIABLE=====" in result.stdout or "=====ERROR=====" in result.stdout):
                #print in red
                print(f"\033[91m \n{instance} FAILED\033[00m")
                errors+=1
        
        totalErrors+=errors
        if (errors==0):
            print("\033[92m Instances passed (elapsed (with overhead) time: "+str(time.time()-t0 )+")\033[00m")

def run_unsat_models(prefix):
    global totalErrors
    for folder in modelsUNSAT:
        folder=prefix+folder
        print("Running UNSAT models in "+folder+" -> ",end="")
        if not os.path.exists(folder):
            print(f"\033[93m FOLDER not found, SKIPPING\033[00m")
            continue
        instances = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
        #forall files in the folder get their name
        #print all instances
        errors=0
        t0 = time.time()
        
        for instance in instances:
            result = subprocess.run(["minizinc", "--solver", solver,folder+instance], capture_output=True, text=True)
            if(not ("=====UNSATISFIABLE====="  in result.stdout) or "=====ERROR=====" in result.stdout):
                #print in red
                print(f"\033[91m \n{instance} FAILED\033[00m")
                errors+=1
        totalErrors+=errors
        if (errors==0):
            print("\033[92m Instances passed (exlapsed (with overhead) time: "+str(time.time()-t0)+")\033[00m")


#start timer


run_sat_models("./SimpleTables/")
run_unsat_models("./SimpleTables/")

run_sat_models("./SmartTables/")
run_unsat_models("./SmartTables/")

if(totalErrors==0):
    print("\n\n\033[92m ************** All instances passed **************\033[00m \n\n")
else:
    print("\n\n \033[91m ************** {totalErrors} instances failed **************\033[00m \n\n")
