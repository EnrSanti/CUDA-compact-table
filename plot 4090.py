import os
import re
import matplotlib.pyplot as plt
import numpy as np  

sys="sys_2"
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
    plt.xticks(x, map(lambda i: i + 1, range(len(seq1))))
    plt.xlabel("Instance no.")
    plt.ylabel("Time (ms)")
    plt.title(sys+", "+type+" tests")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(diagram_name, dpi=300, bbox_inches="tight")  
    print(f"Plot saved as {diagram_name}")

b_serial=[9, 8, 7, 0, 5, 7, 7]
b_ctf=[2, 2, 2, 0, 1, 2, 2]
b_ctuf=[2, 2, 2, 0, 2, 1, 1]
plot_sequences(b_serial,  b_ctf, b_ctuf, "SAT_B.png","B",13)

eb_serial=[ 24, 25, 24, 27, 41]
eb_ctf=[5, 4, 5, 4, 7]
eb_ctuf=[5, 4, 5, 5, 9]
plot_sequences(eb_serial, eb_ctf,eb_ctuf, "SAT_EB.png","EB",55)

eeb_serial=[4, 99, 196, 9, 82, 49, 10, 157, 185, 32, 12, 303, 26, 194, 135, 119]
eeb_ctf=[ 2, 13, 24, 4, 11, 8, 3, 19, 18, 22, 6, 34, 5, 23, 13, 15]
eeb_ctuf=[5, 17, 29, 10, 13, 13, 6, 25, 11, 14, 16, 37, 9, 25, 18, 21]
plot_sequences(eeb_serial, eeb_ctf, eeb_ctuf,  "SAT_EEB.png","EEB",390)
