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

    # Add the "Timeout" legend manually using Line2D for the black bar
    timeout_legend_serial = Line2D([0], [0], color='#faa307', lw=4, label="(Serial) Timeout")
    timeout_legend_uf = Line2D([0], [0], color='#e85d04', lw=4, label="(CUDA CT-uf) Timeout")
    
    for i in range(len(filtered_seq1)):
        if filtered_seq1[i] == 900:
            plt.bar(x[i] - width, 900, width, color='#faa307')  # Black bar at x[i]

    for i in range(len(filtered_seq3)):
        if filtered_seq3[i] == 900:
            plt.bar(x[i] + width, 900, width, color='#e85d04')  # Black bar at x[i]

    if(900 in filtered_seq1):
        if(900 in filtered_seq3):
            plt.legend(handles=[timeout_legend_serial, 
                        timeout_legend_uf,   
                        plt.Line2D([0], [0], color='#b7e4c7', lw=4, label="Serial solve time"),
                        plt.Line2D([0], [0], color='#40916c', lw=4, label="CUDA CT-f solve time"),
                        plt.Line2D([0], [0], color='#1b4332', lw=4, label="CUDA CT-uf solve time")])
        else:
            plt.legend(handles=[timeout_legend_serial, 
                        plt.Line2D([0], [0], color='#e85d04', lw=4, label="CUDA CT-uf Timeout"),
                        plt.Line2D([0], [0], color='#b7e4c7', lw=4, label="Serial solve time"),
                        plt.Line2D([0], [0], color='#40916c', lw=4, label="CUDA CT-f solve time")])
    else:
        print(filtered_seq3)
        if(900 in filtered_seq3):
            plt.legend(handles=[timeout_legend_uf,   
                        plt.Line2D([0], [0], color='#b7e4c7', lw=4, label="Serial solve time"),
                        plt.Line2D([0], [0], color='#40916c', lw=4, label="CUDA CT-f solve time"),
                        plt.Line2D([0], [0], color='#1b4332', lw=4, label="CUDA CT-uf solve time")])
        else:
            plt.legend(handles=[plt.Line2D([0], [0], color='#b7e4c7', lw=4, label="Serial solve time"),
                        plt.Line2D([0], [0], color='#40916c', lw=4, label="CUDA CT-f solve time"),
                        plt.Line2D([0], [0], color='#1b4332', lw=4, label="CUDA CT-uf solve time")])
    # Set the x-ticks to the equidistant positions but label them with the actual indexes
    plt.xticks(x, indexes)  
    plt.xlabel("Instance no.")
    plt.ylabel("Time (s)")
    plt.title(sys+", "+type+" tests")
    plt.ylim(0, maxVal)
    #use normal scale
    plt.yscale("linear")
    # Set the y-axis limit to avoid negative values
    plt.ylim(bottom=0)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(diagram_name, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {diagram_name}")


b_serial=["Timeout", "Timeout", "Timeout", "Timeout", 429.71, "Timeout", "Timeout", 2.27, 226.79, 7.58, "Timeout", 300.15, 9.7, "Timeout", "Timeout", 3.43, "Timeout", "Timeout", 0, 427.8, 13.33, "Timeout", "Timeout", "Timeout", "Timeout", 8.71, 0, 3.59, "Timeout", "Timeout", 0, 8.73, "Timeout", "Timeout", 5.79, 837.51, 545.67, "Timeout", 12.05, 2.15]
b_ctf=["Timeout", "Timeout", "Timeout", "Timeout", 222.921, "Timeout", "Timeout", 1.44, 104.93, 3.93, "Timeout", 216.25, 4.25, "Timeout", "Timeout", 1.77, "Timeout", "Timeout", 0, 278.33, 5.62, 653.7, "Timeout", "Timeout", 470.87, 4.21, 0, 1.31, "Timeout", "Timeout", 0, 4.3, 616.32, "Timeout", 3.32, 449.05, 302.91, "Timeout", 5.74, 2.08]
b_ctuf=["Timeout", "Timeout", "Timeout", "Timeout", 205.69, "Timeout", "Timeout", 1.36, 86, 3.72, "Timeout", 197.3, 3.76, "Timeout", "Timeout", 1.69, "Timeout", "Timeout", 0, 257.14, 5.13, 570.65, "Timeout", "Timeout", 418.47, 3.76, 0, 0.98, "Timeout", "Timeout", 0, 3.96, 519.7, "Timeout", 3.08, 427.53, 270.32, "Timeout", 5.22, 1.6]
plot_sequences(b_serial,  b_ctf, b_ctuf, "lin_b.png","B",950)

eb_serial=[2.17, "Timeout", "Timeout", 80.55, 181.43, "Timeout", 0.47, 2.07, "Timeout", 1.34, "Timeout", 181.32, "Timeout", 1.44, 1.14, "Timeout", 385.71, "Timeout", "Timeout", 1.26, 1.78, 2.6, 3.73, "Timeout", 0.32, "Timeout", "Timeout", 0, 342.51, "Timeout", 0.78, 204.92, "Timeout", "Timeout", "Timeout", 0.4, 0.28, 2.41, 1.55, 0.56]
eb_ctf=[2.17, "Timeout", "Timeout", 34.34, 125.7, "Timeout", 0.51, 1.52, "Timeout", 0.76, "Timeout", 118.224, "Timeout", 0.91, 0.9, "Timeout", 209.13, "Timeout", "Timeout", 0.9, 1.03, 1.35, 1.96, "Timeout", 0.25, "Timeout", "Timeout", 0, 161.94, "Timeout", 0.78, 148.49, "Timeout", "Timeout", "Timeout", 0.34, 0.35, 1.56, 1.27, 0.8]
eb_ctuf=[2.16, "Timeout", "Timeout", 28.23, 111.39, "Timeout", 0.45, 1.37, "Timeout", 0.67, "Timeout", 103.57, "Timeout", 0.82, 0.81, "Timeout", 185.42, "Timeout", "Timeout", 0.82, 0.93, 1.16, 1.8, "Timeout", 0.23, "Timeout", "Timeout", 0, 142.27, "Timeout", 0.68, 132.83, "Timeout", "Timeout", "Timeout", 0.28, 0.3, 1.44, 0.97, 0.64]
plot_sequences(eb_serial, eb_ctf,eb_ctuf, "lin_eb.png","EB",550)

