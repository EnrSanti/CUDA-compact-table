import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

files=["./4090_results_f_uf_5_eb.ods","./4090_results_f_uf_6_b.ods"]    
varsNames=["eb_serial_5","eb_ctf_5","eb_ctuf_5","b_serial_6","b_ctf_6","b_ctuf_6"]
def getData(models):
    global varsNames
    for folderIndex in range(0,len(models),1):
        

        _file=models[folderIndex]

        df = pd.read_excel(_file, engine="odf")


        serial=[]
        CU_uf=[]
        CU_f=[]
        
        testNos=[]
        testNo=1
        # Iterate through each row
        for index, row in df.iterrows():
            #check if first column of the considered row is nan (empty)
            if(pd.isnull(row[df.columns[0]])):
                #just a placeholder for the test number
                serial.append("Timeout")
                CU_f.append("Timeout")
                CU_uf.append("Timeout")
            else:
                if(row[df.columns[1]]!="Timeout"):
                    serial.append(float(row[df.columns[1]]))
                else:
                    serial.append("Timeout")
                if(row[df.columns[2]]!="Timeout"):
                    CU_f.append(float(row[df.columns[2]]))
                else:
                    CU_f.append("Timeout")
                if(row[df.columns[3]]!="Timeout"):
                    CU_uf.append(float(row[df.columns[3]]))
                else:
                    CU_uf.append("Timeout")

        print(varsNames[folderIndex*3]+"="+str(serial))
        print(varsNames[folderIndex*3+1]+"="+str(CU_f))
        print(varsNames[folderIndex*3+2]+"="+str(CU_uf))

getData(files)