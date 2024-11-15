import os
import subprocess

# List of model files
modelsSAT = ["serial/SAT/",
"CUDA/SAT/",
"TestGenerator & more tests/testsSAT/",
"TestGenerator & more tests/testsSAT_CUDA/"]

modelsUNSAT = ["serial/UNSAT/",
"CUDA/UNSAT/",
"TestGenerator & more tests/testsUNSAT/",
"TestGenerator & more tests/testsUNSAT_CUDA/"]

solver="MiniCpp"
# Run each serial SAT model
print("Running simple models")

totalErrors=0

def run_sat_models(prefix):
    global totalErrors
    for folder in modelsSAT:  
        folder=prefix+folder
        print("Running SAT models in "+folder+" -> ",end="")

        instances = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
        #forall files in the folder get their name
        #print all instances
        errors=0
        for instance in instances:
            result = subprocess.run(["minizinc", "--solver", solver,folder+instance], capture_output=True, text=True)
            if("=====UNSATISFIABLE=====" in result.stdout):
                #print in red
                print(f"\033[91m \n{instance} FAILED\033[00m")
                errors+=1
        totalErrors+=errors
        if (errors==0):
            print("\033[92m Instances passed\033[00m")

def run_unsat_models(prefix):
    global totalErrors
    for folder in modelsUNSAT:
        folder=prefix+folder
        print("Running UNSAT models in "+folder+" -> ",end="")

        instances = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
        #forall files in the folder get their name
        #print all instances
        errors=0
        for instance in instances:
            result = subprocess.run(["minizinc", "--solver", solver,folder+instance], capture_output=True, text=True)
            if(not ("=====UNSATISFIABLE=====" in result.stdout)):
                #print in red
                print(f"\033[91m \n{instance} FAILED\033[00m")
                errors+=1
        totalErrors+=errors
        if (errors==0):
            print("\033[92m Instances passed\033[00m")



run_sat_models("./SimpleTables/")
run_unsat_models("./SimpleTables/")
run_sat_models("./SmartTables/")
run_unsat_models("./SmartTables/")

if(totalErrors==0):
    print("\n\n\033[92m ************** All instances passed **************\033[00m \n\n")
else:
    print("\n\n \033[91m ************** {totalErrors} instances failed **************\033[00m \n\n")
