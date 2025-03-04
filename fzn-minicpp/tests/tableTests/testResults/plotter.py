import os
import re
import matplotlib.pyplot as plt
import numpy as np  

# List of model files

folderSAT = [
"./TestGenerator & more tests/testsSAT_CUDA_bigger/",
"./TestGenerator & more tests/testsSAT_CUDA_even_bigger/",
"./TestGenerator & more tests/testsSAT_CUDA_even_even_bigger/"]

setNamesSAT = ["SAT_B","SAT_EB","SAT_EEB"]    

def plots(models):

    for folderIndex in range(0,len(models),1):
        
        serialFolderTime=0
        cudaFolderTime=0

        folder=models[folderIndex]

        if not os.path.exists(folder):
            print(f"\033[93m FOLDER {folder} not found, SKIPPING\033[00m")
            continue
        
        instances = sorted(   [file for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))], key=extract_number)



        
        #check if folder exists, print in yellow

        cudaTimes=[]
        serialTimes=[]

        for instance in instances:
            #read instance file 
            print(f"Reading {instance}")
            with open(folder+instance, "r") as text_file:
                time_serial,time_cuda = filter_input(text_file.read())
                print(f"{time_serial} {time_cuda} percentage speedup {percentage_speedup(time_serial, time_cuda):.2f}%")
                serialTimes.append(time_serial)
                cudaTimes.append(time_cuda)


        plot_sequences(serialTimes, cudaTimes,setNamesSAT[folderIndex])
    

def percentage_speedup(old_time, new_time):
    if(old_time == 0):
        return -1
    speedup = ((old_time - new_time) / old_time) * 100
    return speedup

def extract_number(filename):
    match = re.search(r'\d+', filename)  # Find the first number in the filename
    return int(match.group()) if match else float('inf')  # Convert to int


def filter_input(input_string):
    lines = input_string.splitlines()  # Split string into lines
    filtered = [line for line in lines if "%%%mzn-stat: solveTime" in line]
    return float(re.search(r'\d+\.\d{3}', filtered[0]).group()),float(re.search(r'\d+\.\d{3}', filtered[1]).group())  # SERIAL AND CUDA TIMES

def plot_sequences(seq1, seq2, diagram_name):
    x = np.arange(len(seq1))  # Generate x positions
    width = 0.4  # Width of bars
    
    plt.figure(figsize=(10, 5))
    
    plt.bar(x - width/2, seq1, width, color='#b7e4c7', label='Serial solve time')
    plt.bar(x + width/2, seq2, width, color='#40916c', label='CUDA solve time')
    
    # Show only INTEGER NUMBERS on the X axis
    plt.xticks(x, range(0, len(seq1)))
    plt.xlabel("Instance no.")
    plt.ylabel("Time (s)")
    plt.title(diagram_name)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(diagram_name, dpi=300, bbox_inches="tight")  
    print(f"Plot saved as {diagram_name}")


plots(folderSAT)
#plots(folderUNSAT)


print("\n\n\033[92m ************** DATA GATHERED **************\033[00m \n\n")
