import os
import subprocess
import time

#it just check sat/unsat (on the small instances you can also check the assignments)

# List of model files
modelsSAT = [
"serial/SAT/",
"CUDA/SAT/",
"TestGenerator & more tests/testsSAT/",
"TestGenerator & more tests/testsSAT_CUDA/",
"TestGenerator & more tests/testsSAT_bigger/",
"TestGenerator & more tests/testsSAT_CUDA_bigger/",
"TestGenerator & more tests/testsSAT_even_bigger/",
"TestGenerator & more tests/testsSAT_CUDA_even_bigger/",
"TestGenerator & more tests/testsSAT_even_even_bigger/",
"TestGenerator & more tests/testsSAT_CUDA_even_even_bigger/"
]

modelsUNSAT = [
"serial/UNSAT/",
"CUDA/UNSAT/",
"TestGenerator & more tests/testsUNSAT/",
"TestGenerator & more tests/testsUNSAT_CUDA/",
"TestGenerator & more tests/testsUNSAT_bigger/",
"TestGenerator & more tests/testsUNSAT_CUDA_bigger/",
"TestGenerator & more tests/testsUNSAT_even_bigger/",
"TestGenerator & more tests/testsUNSAT_CUDA_even_bigger/",
"TestGenerator & more tests/testsUNSAT_even_even_bigger/",
"TestGenerator & more tests/testsUNSAT_CUDA_even_even_bigger/"
]

solver="MiniCpp"
# Run each serial SAT model
print("Running simple models")

totalErrors=0

def run_sat_models(prefix):
    global totalErrors
    resultsSAT=""

    for folderIndex in range(0,len(modelsSAT),2):
        
        serialFolderTime=0
        cudaFolderTime=0

        folder=prefix+modelsSAT[folderIndex]
        folderCUDA=prefix+modelsSAT[folderIndex+1]
        

        print("Comparing "+folder+" and "+ folderCUDA)

        if not os.path.exists(folder):
            print(f"\033[93m FOLDER not found, SKIPPING\033[00m")
            continue
        
        instances = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]

        #forall files in the folder get their name
        #print all instances
        errors=0
        
        #check if folder exists, print in yellow
        for instance in instances:

            t0_s = time.time()
            result = subprocess.run(["minizinc", "--solver", solver,folder+instance], capture_output=True, text=True)
            t1_s=time.time()

            if("=====UNSATISFIABLE=====" in result.stdout or "=====ERROR=====" in result.stdout):
                #print in red
                print(f"\033[91m \n{folder+instance} FAILED\033[00m")
                errors+=1

            if os.path.exists(os.path.join(folderCUDA, instance)):
                t0_CUDA = time.time()
                resultCUDA = subprocess.run(["minizinc", "--solver", solver,folderCUDA+instance], capture_output=True, text=True)
                t1_CUDA=time.time()
                
                delta_s = t1_s-t0_s
                delta_CUDA = t1_CUDA-t0_CUDA

                if("=====UNSATISFIABLE=====" in resultCUDA.stdout or "=====ERROR=====" in resultCUDA.stdout):
                    #print in red
                    print(f"\033[91m \n{folderCUDA+instance} FAILED\033[00m")
                    errors+=1
                    continue
                
                if(result.stdout[:-1]!=resultCUDA.stdout[:-1]):
                    print(f"\033[91m {instance} RESULT MISMATCH\033[00m")
                    errors+=1

                print("\033[92m"+instance+", SERIAL: "+str(delta_s)+", CUDA: "+str(delta_CUDA)+ " \033[00m")  
                cudaFolderTime+=delta_CUDA
                serialFolderTime+=delta_s
            
            else:
                print(f"\033[93mmatching file not found for "+str(instance)+", SKIPPING\033[00m")
                continue
            
        if (errors==0):
            print("\033[92m\n\nInstances passed SERIAL: "+str(serialFolderTime)+", CUDA: "+str(cudaFolderTime)+ "\033[00m")
            resultsSAT+="SERIAL ("+folder+"): "+str(serialFolderTime)+", CUDA: "+str(cudaFolderTime)+ "\n"
        print("-------------------------------------------------------------------------------------------\n")
    return resultsSAT
def run_unsat_models(prefix):
    global totalErrors
    resultsUNSAT=""

    for folderIndex in range(0,len(modelsUNSAT),2):

        
        serialFolderTime=0
        cudaFolderTime=0


        folder=prefix+modelsUNSAT[folderIndex]
        folderCUDA=prefix+modelsUNSAT[folderIndex+1]
        

        print("Comparing "+folder+" and "+ folderCUDA)

        if not os.path.exists(folder):
            print(f"\033[93m FOLDER not found, SKIPPING\033[00m")
            continue
        
        instances = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]

        #forall files in the folder get their name
        #print all instances
        errors=0
        
        #check if folder exists, print in yellow
        for instance in instances:

            t0_s = time.time()
            result = subprocess.run(["minizinc", "--solver", solver,folder+instance], capture_output=True, text=True)
            t1_s=time.time()

            if(not ("=====UNSATISFIABLE====="  in result.stdout) or "=====ERROR=====" in result.stdout):
                #print in red
                print(f"\033[91m \n{folder+instance} FAILED\033[00m")
                errors+=1

            if os.path.exists(os.path.join(folderCUDA, instance)):
                t0_CUDA = time.time()
                resultCUDA = subprocess.run(["minizinc", "--solver", solver,folderCUDA+instance], capture_output=True, text=True)
                t1_CUDA=time.time()

                delta_s = t1_s-t0_s
                delta_CUDA = t1_CUDA-t0_CUDA
                if(not ("=====UNSATISFIABLE====="  in result.stdout) or "=====ERROR=====" in result.stdout):
                    #print in red
                    print(f"\033[91m \n{folderCUDA+instance} FAILED\033[00m")
                    errors+=1
                    continue
                
                if(result.stdout[:-1]!=resultCUDA.stdout[:-1]):
                    print(f"\033[91m {instance} RESULT MISMATCH\033[00m")
                    errors+=1

                print("\033[92m"+instance+", SERIAL: "+str(delta_s)+", CUDA: "+str(delta_CUDA)+ " \033[00m")  
                cudaFolderTime+=(delta_CUDA)
                serialFolderTime+=(delta_s)
            
            else:
                print(f"\033[93mmatching file not found for "+str(instance)+", SKIPPING\033[00m")
                
                continue
            
        if (errors==0):
            print("\033[92m\n\nInstances passed SERIAL: "+str(serialFolderTime)+", CUDA: "+str(cudaFolderTime)+ "\033[00m")
            resultsUNSAT+="SERIAL ("+folder+"): "+str(serialFolderTime)+", CUDA: "+str(cudaFolderTime)+ "\n"
 

        print("-------------------------------------------------------------------------------------------\n")
    return resultsUNSAT

resultsSAT=run_sat_models("./SimpleTables/")
resultsUNSAT=run_unsat_models("./SimpleTables/")


#resultsSAT_smart=run_sat_models("./SmartTables/")
#resultsUNSAT_smart=run_unsat_models("./SmartTables/")


if(totalErrors==0):
    print("\n\n\033[92m ************** All instances passed **************\033[00m \n\n")
    print("SAT recap:\n")
    print(resultsSAT)
    print("UNSAT recap:\n")
    print(resultsUNSAT)
    print("-------------------\n")
    print("Smart SAT recap:\n")
    print(resultsSAT_smart)
    print("Smart UNSAT recap:\n")
    print(resultsUNSAT_smart)

else:
    print("\n\n \033[91m ************** {totalErrors} instances failed **************\033[00m \n\n")
