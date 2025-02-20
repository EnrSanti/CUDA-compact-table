import os
import re
import matplotlib.pyplot as plt
import numpy as np  

sys="sys_1"
def plot_sequences(seq1, seq2, seq3, diagram_name, type,maxVal):
    global sys
    x = np.arange(len(seq1))  # Generate x positions
    width = 0.2  # Adjust width to fit 3 bars in a group

    plt.figure(figsize=(12, 6))

    # Shifting the bars properly to align them in a group
    plt.bar(x - width, seq1, width, color='#b7e4c7', label='Serial solve time')
    plt.bar(x, seq2, width, color='#40916c', label='CUDA CT-f solve time')
    plt.bar(x + width, seq3, width, color='#1b4332', label='CUDA CT-uf solve time')

    # Show only INTEGER NUMBERS on the X axis
    plt.ylim(0, maxVal)
    plt.xticks(x, range(len(seq1)))
    plt.xlabel("Instance no.")
    plt.ylabel("Time (ms)")
    plt.title(sys+", "+type+" tests")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(diagram_name, dpi=300, bbox_inches="tight")  
    print(f"Plot saved as {diagram_name}")

b_serial=[12, 10, 10, 0, 6, 8, 10]
b_ctf=[3, 3, 3, 0, 2, 2, 2]
b_ctuf=[4, 3, 3, 1, 3, 3, 3]
plot_sequences(b_serial,  b_ctf, b_ctuf, "SAT_B.png","B",13)

eb_serial=[31, 33, 30, 38, 52]
eb_ctf=[ 7, 6, 7, 6, 10]
eb_ctuf=[10, 9, 10, 10, 19]
plot_sequences(eb_serial, eb_ctf,eb_ctuf, "SAT_EB.png","EB",55)

eeb_serial=[5, 132, 239, 11, 100, 58, 12, 189, 1, 1, 14, 374, 31, 239, 165, 145]
eeb_ctf=[3, 20, 35, 5, 16, 12, 4, 28, 1, 1, 8, 54, 7, 35, 21, 23]
eeb_ctuf=[ 15, 47, 87, 28, 33, 35, 15, 66, 1, 1, 45, 98, 21, 66, 49, 52]
plot_sequences(eeb_serial, eeb_ctf, eeb_ctuf,  "SAT_EEB.png","EEB",390)
