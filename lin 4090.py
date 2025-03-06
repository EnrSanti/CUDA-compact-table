import os
import re
import matplotlib.pyplot as plt
import numpy as np  
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd

sys="sys_2"
def plot_sequences(seq1, seq2, seq3, diagram_name, type,maxVal):    
    # Get all indexes where at least one sequence is not "Timeout"
    indexes = sorted(set(
        index for index, value in enumerate(seq1) if value != "Timeout"
    ) | set(
        index for index, value in enumerate(seq2) if value != "Timeout"
    ) | set(
        index for index, value in enumerate(seq3) if value != "Timeout"
    ))

    # Replace "Timeout" values with 900
    seq1 = [900 if value == "Timeout" else value for value in seq1]
    seq2 = [900 if value == "Timeout" else value for value in seq2]
    seq3 = [900 if value == "Timeout" else value for value in seq3]

    # Extract only the values at the selected indexes
    filtered_seq1 = [seq1[i] for i in indexes]
    filtered_seq2 = [seq2[i] for i in indexes]
    filtered_seq3 = [seq3[i] for i in indexes]

    x = np.arange(len(indexes))*1.6  # Create equidistant x positions
    width = 0.4  # Width of bars

    plt.figure(figsize=(10, 5))

    # Plot only filtered values with equidistant x positions
    plt.bar(x - width, filtered_seq1, width, color='#b7e4c7')
    plt.bar(x, filtered_seq2, width, color='#40916c')
    plt.bar(x + width, filtered_seq3, width, color='#1b4332')

    
    for i in range(len(filtered_seq1)):
        if filtered_seq1[i] == 900:
            plt.bar(x[i] - width, 900, width, color='#b7e4c7')  # Black bar at x[i]

    for i in range(len(filtered_seq3)):
        if filtered_seq3[i] == 900:
            plt.bar(x[i] + width, 900, width, color='#1b4332')  # Black bar at x[i]

    plt.legend(handles=[plt.Line2D([0], [0], color='#b7e4c7', lw=4, label="Serial solve time"),
                plt.Line2D([0], [0], color='#40916c', lw=4, label="CUDA CT-f solve time"),
                plt.Line2D([0], [0], color='#1b4332', lw=4, label="CUDA CT-uf solve time")])

    # Set the x-ticks to the equidistant positions but label them with the actual indexes
    plt.xticks(x, map(lambda i: i + 1, indexes))  
    plt.xlabel("Instance no.")
    plt.ylabel("Time (s)")
    plt.title(sys+", "+type+" tests")
    plt.ylim(0, maxVal)
    #use normal scale
    plt.yscale("linear")
    # Set the y-axis limit to avoid negative values
    plt.ylim(bottom=0)

    # Add a red dashed line at y = 900
    if(maxVal>890):
        plt.axhline(y=900, color='red', linestyle='dashed', linewidth=1)

        # Add a "Timeout" label near the line
        plt.text(x[-2] + width, 900, "Timeout", color='red', fontsize=10, verticalalignment='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(diagram_name, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {diagram_name}")

eb_serial_5=[681.55, 'Timeout', 'Timeout', 'Timeout', 429.71, 'Timeout', 'Timeout', 157.09, 226.79, 243.0, 'Timeout', 300.15, 410.16, 'Timeout', 'Timeout', 656.65, 'Timeout', 'Timeout', 293.59, 427.8, 13.33, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 12.21, 617.58, 168.85, 'Timeout', 'Timeout', 'Timeout', 16.62, 'Timeout', 'Timeout', 84.06, 837.51, 545.67, 'Timeout', 12.05, 366.06]
eb_ctf_5=[385.89, 'Timeout', 'Timeout', 'Timeout', 222.921, 'Timeout', 'Timeout', 29.11, 104.93, 61.48, 'Timeout', 216.25, 287.36, 'Timeout', 'Timeout', 560.87, 'Timeout', 'Timeout', 233.214, 278.33, 5.62, 653.7, 'Timeout', 'Timeout', 470.87, 4.8, 392.05, 31.84, 'Timeout', 'Timeout', 452.51, 6.41, 616.32, 'Timeout', 26.42, 449.05, 302.91, 'Timeout', 5.74, 251.15]
eb_ctuf_5=[335.84, 'Timeout', 'Timeout', 'Timeout', 205.69, 'Timeout', 'Timeout', 11.78, 86.0, 41.04, 'Timeout', 197.3, 258.75, 'Timeout', 'Timeout', 445.27, 'Timeout', 'Timeout', 201.994, 257.14, 5.13, 570.65, 'Timeout', 'Timeout', 418.47, 4.16, 358.68, 22.13, 'Timeout', 'Timeout', 416.57, 5.61, 519.7, 'Timeout', 10.15, 427.53, 270.32, 'Timeout', 5.22, 224.67]
b_serial_6=[344.56, 119.124, 'Timeout', 80.55, 181.43, 'Timeout', 150.76, 24.54, 35.53, 145.47, 'Timeout', 181.32, 256.94, 140.69, 192.49, 'Timeout', 385.71, 'Timeout', 'Timeout', 290.37, 219.63, 342.9, 85.71, 'Timeout', 32.75, 'Timeout', 'Timeout', 269.83, 342.51, 'Timeout', 160.57, 204.92, 'Timeout', 'Timeout', 'Timeout', 370.91, 362.84, 220.15, 380.49, 229.6]
b_ctf_6=[184.56, 78.051, 'Timeout', 34.34, 125.7, 'Timeout', 97.03, 5.96, 17.21, 125.46, 'Timeout', 118.224, 171.25, 118.15, 120.93, 'Timeout', 209.13, 'Timeout', 'Timeout', 192.61, 132.94, 189.62, 74.67, 'Timeout', 13.62, 'Timeout', 'Timeout', 133.21, 161.94, 'Timeout', 80.38, 148.49, 'Timeout', 'Timeout', 'Timeout', 177.8, 185.69, 172.08, 180.88, 149.0]
b_ctuf_6=[147.06, 64.735, 'Timeout', 28.23, 111.39, 'Timeout', 81.5, 3.9, 13.98, 107.98, 'Timeout', 103.57, 152.529, 99.08, 104.26, 'Timeout', 185.42, 'Timeout', 'Timeout', 168.86, 116.24, 163.02, 64.32, 'Timeout', 10.11, 'Timeout', 'Timeout', 111.18, 142.27, 'Timeout', 69.49, 132.83, 'Timeout', 'Timeout', 'Timeout', 149.35, 163.078, 144.276, 151.671, 142.337]


plot_sequences(eb_serial_5,  eb_ctf_5, eb_ctuf_5, "lin_eb_4090.png","EB",950)
plot_sequences(b_serial_6, b_ctf_6,b_ctuf_6, "lin_b_4090.png","B",550)

