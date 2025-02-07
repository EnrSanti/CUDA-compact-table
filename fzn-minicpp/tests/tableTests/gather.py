import os
import subprocess
import time

#it just check sat/unsat (on the small instances you can also check the assignments)

# List of model files
modelsSAT = [
"TestGenerator & more tests/testsSAT_bigger/",
"TestGenerator & more tests/testsSAT_CUDA_bigger/",
"TestGenerator & more tests/testsSAT_even_bigger/",
"TestGenerator & more tests/testsSAT_CUDA_even_bigger/",
"TestGenerator & more tests/testsSAT_even_even_bigger/",
"TestGenerator & more tests/testsSAT_CUDA_even_even_bigger/"]


modelsUNSAT = [
"TestGenerator & more tests/testsUNSAT_bigger/",
"TestGenerator & more tests/testsUNSAT_CUDA_bigger/",
"TestGenerator & more tests/testsUNSAT_even_bigger/",
"TestGenerator & more tests/testsUNSAT_CUDA_even_bigger/",
"TestGenerator & more tests/testsUNSAT_even_even_bigger/",
"TestGenerator & more tests/testsUNSAT_CUDA_even_even_bigger/"]

modelsFromCSP = [
"./TestGenerator & more tests/VersoLeTable/Botti/",
"./TestGenerator & more tests/VersoLeTable/Inst0/",
"./TestGenerator & more tests/VersoLeTable/Inst1/"]

solver="MiniCpp"
# Run each serial SAT model
print("Gathering times")

totalErrors=0

def run_models(prefix,models):

    for folderIndex in range(0,len(models),2):
        
        serialFolderTime=0
        cudaFolderTime=0

        folder=prefix+models[folderIndex]
        folderCUDA=prefix+models[folderIndex+1]

        print("Comparing "+folder+" and "+ folderCUDA)

        if not os.path.exists(folder):
            print(f"\033[93m FOLDER not found, SKIPPING\033[00m")
            continue
        
        instances = [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]

        
        #check if folder exists, print in yellow
        for instance in instances:

            result = subprocess.run(["minizinc",  "--solver", solver, "--statistics" ,folder+instance], capture_output=True, text=True)

            setialOutput=filter_output(result.stdout)

            if os.path.exists(os.path.join(folderCUDA, instance)):
                resultCUDA = subprocess.run(["minizinc", "--solver",solver,"--statistics" ,folderCUDA+instance], capture_output=True, text=True)
                cudaOutput=filter_output(resultCUDA.stdout)
              
                os.makedirs("./testResults/"+models[folderIndex+1], exist_ok=True)  # Creates the folder if it doesn't exist

                #create a new file under modelsSAT[folderIndex] with the output
                with open("./testResults/"+models[folderIndex+1]+instance[:-3]+"out", "w") as text_file:
                    text_file.write(setialOutput+"\n *************** \n"+cudaOutput)
                    print("Output written to "+models[folderIndex+1]+instance[:-3]+"out")
            else:
                print(f"\033[93 matching file not found for "+str(instance)+", SKIPPING\033[00m")
                continue
            
        print("\033[92m\n FOLDERs DONE \033[00m")
        print("-------------------------------------------------------------------------------------------\n")



def filter_output(input_string):
    lines = input_string.splitlines()  # Split string into lines
    filtered = [line for line in lines if line.startswith("%%% Time") or "%%%mzn-stat: solveTime" in line]
    return "\n".join(filtered)  # Join filtered lines back into a string




# ------------------------------------------------- MAIN -------------------------------------------------
 
#run_models("./SimpleTables/",modelsSAT)
#run_models("./SimpleTables/",modelsUNSAT)
#print("\n\n\033[92m ************** Synthetic instances GATHERED **************\033[00m \n\n")

#gather the data from CSP instances

#for all csp folders
i=0
for folder in modelsFromCSP:

    folder="./SimpleTables/"+folder
    if not os.path.exists(folder):
        print(f"\033[93m FOLDER {folder} not found, SKIPPING\033[00m")
        continue
    
    serial_instances = [file for file in os.listdir(folder) 
             if os.path.isfile(os.path.join(folder, file)) and not file.startswith("CUDA")]

    for instance in serial_instances:

        result = subprocess.run(["minizinc",  "--solver", solver, "--statistics" ,folder+instance], capture_output=True, text=True)

        setialOutput=filter_output(result.stdout)
        print(instance)
    
        if os.path.exists(os.path.join(folder, "CUDA_"+instance)):
            resultCUDA = subprocess.run(["minizinc", "--solver",solver,"--statistics" ,folder+"CUDA_"+instance], capture_output=True, text=True)
            cudaOutput=filter_output(resultCUDA.stdout)
            print("saving to "+"./testResults/"+folder)
            os.makedirs("./testResults/"+folder, exist_ok=True)
            #create a new file under modelsSAT[folderIndex] with the output
            with open("./testResults/"+folder+instance[:-3]+"out", "w") as text_file:
                text_file.write(setialOutput+"\n *************** \n"+cudaOutput)
                print("Output written to "+modelsFromCSP[i]+instance[:-3]+"out")
        else:
            print("\033[93 matching file not printfound for "+str(instance)+"looking for "+str(os.path.join(folder, "CUDA_"+instance))+" SKIPPING\033[00m")
            continue
    i+=1
