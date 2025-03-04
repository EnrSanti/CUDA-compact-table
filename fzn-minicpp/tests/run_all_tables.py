import os
import subprocess
import time

#it just check sat/unsat (on the small instances you can also check the assignments)

# List of model files
modelsSAT = [
"oneSolSAT_10_50_500/",
"oneSolSAT_CUDA_10_50_500/",
"oneSolSAT_10_100_1000/",  #domains 100
"oneSolSAT_CUDA_10_100_1000/",
"oneSolSAT_15_100_1000/",
"oneSolSAT_CUDA_15_100_1000/",
"oneSolSAT_20_80_2500/",
"oneSolSAT_CUDA_20_80_2500/"
]



solver="MiniCpp"
# Run each serial SAT model
print("Running simple models")

totalErrors=0

def get_lines_range(text):
    lines = text.splitlines()
    
    # Remove lines starting with "%%%mzn-stat: flatTime"
    filtered_lines = [line for line in lines if not line.startswith("%%%mzn-stat: flatTime")]
    
    # Extract the range from 7th line to the 13th from the end
    return "\n".join(filtered_lines[6:-13]) if len(filtered_lines) > 19 else ""


def filter_output(input_string):
    lines = input_string.splitlines()
    filtered = [line for line in lines if line.startswith("%%%mzn-stat: propagations")]
    return "\n".join(filtered)


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
            result = subprocess.run(["minizinc", "--solver", solver,"--statistics",folder+instance], capture_output=True, text=True)
            t1_s=time.time()

            if("=====UNSATISFIABLE=====" in result.stdout or "=====ERROR=====" in result.stdout):
                #print in red
                print(f"\033[91m \n{folder+instance} FAILED\033[00m")
                errors+=1

            if os.path.exists(os.path.join(folderCUDA, instance)):
                t0_CUDA = time.time()
                resultCUDA = subprocess.run(["minizinc", "--solver" ,solver,"--statistics",folderCUDA+instance], capture_output=True, text=True)
                t1_CUDA=time.time()
                
                delta_s = t1_s-t0_s
                delta_CUDA = t1_CUDA-t0_CUDA


                if("=====UNSATISFIABLE=====" in resultCUDA.stdout or "=====ERROR=====" in resultCUDA.stdout):
                    #print in red
                    print(f"\033[91m \n{folderCUDA+instance} FAILED\033[00m")
                    errors+=1
                    continue

                cudaLines=get_lines_range(resultCUDA.stdout)
                serialLines=get_lines_range(result.stdout)


                if(cudaLines!=serialLines):
                    print(f"\033[91m {instance} RESULT MISMATCH\033[00m")
                    print(f"\033[91m {instance} CUDA: \n{cudaLines}\033[00m")
                    print(f"\033[91m {instance} SERIAL: \n{serialLines}\033[00m")   
                    errors+=1   

                if(filter_output(result.stdout)!=filter_output(resultCUDA.stdout)):
                    print(f"\033[91m {instance} PROPAGATION NO MISMATCH \033[00m")
                    errors+=1
                print(filter_output(result.stdout)+" "+filter_output(resultCUDA.stdout))

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
resultsSAT=run_sat_models("./")


if(totalErrors==0):
    print("\n\n\033[92m ************** All instances passed **************\033[00m \n\n")
    print("SAT recap:\n")
    print(resultsSAT)

else:
    print("\n\n \033[91m ************** {totalErrors} instances failed **************\033[00m \n\n")
