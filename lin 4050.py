import os
import re
import matplotlib.pyplot as plt
import numpy as np  
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd

sys="sys_1"
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


b_serial=["Timeout", "Timeout", "Timeout", "Timeout", 562.93, "Timeout", "Timeout", 2.94, 309.33, 9.69, "Timeout", 385.05, 12.57, "Timeout", "Timeout", 4.45, "Timeout", "Timeout", 0, 565.19, 17.66, "Timeout", "Timeout", "Timeout", "Timeout", 11.08, 0, 5.03, "Timeout", "Timeout", 0, 11.15, "Timeout", "Timeout", 7.4, "Timeout", 716.89, "Timeout", 15.58, 2.78]
b_ctf=["Timeout", "Timeout", "Timeout", "Timeout", 274.29, "Timeout", "Timeout", 1.67, 116.02, 4.47, "Timeout", 248.52, 4.57, "Timeout", "Timeout", 2.21, "Timeout", "Timeout", 0, 331.85, 6.1, 678.88, "Timeout", "Timeout", 472.67, 0, 0, 1.52, "Timeout", "Timeout", 0, 4.57, 645.92, "Timeout", 3.66, 558.71, 355.14, "Timeout", 5.8, 1.82]
b_ctuf=["Timeout", "Timeout", "Timeout", "Timeout", 329.93, "Timeout", "Timeout", 2.12, 139.83, 6.93, "Timeout", 296.01, 6.61, "Timeout", "Timeout", 2.8, "Timeout", "Timeout", 0, 412.69, 9.78, "Timeout", "Timeout", "Timeout", 668.18, 6.61, 0, 1.85, "Timeout", "Timeout", 0, 7.17, 868.07, "Timeout", 5.42, 740.11, 439.13, "Timeout", 10.12, 2.52]
plot_sequences(b_serial,  b_ctf, b_ctuf, "lin_b.png","B",950)

eb_serial=[2.77, "Timeout", "Timeout", 105.21, 237.75, "Timeout", 0.59, 2.66, "Timeout", 1.73, "Timeout", 229.95, "Timeout", 1.83, 1.47, "Timeout", 499.76, "Timeout", "Timeout", 1.63, 2.26, 3.37, 5.13, "Timeout", 0.42, "Timeout", "Timeout", 0, 440.91, "Timeout", 0.97, 260.41, "Timeout", "Timeout", "Timeout", 0.51, 0.34, 3.09, 2.1, 0.69]
eb_ctf=[2.87, "Timeout", "Timeout", 43.85, 136.47, "Timeout", 0.51, 1.62, "Timeout", 0.91, "Timeout", 132.99, "Timeout", 1.08, 0.99, "Timeout", 230.08, "Timeout", "Timeout", 1.09, 1.21, 1.61, 2.25, "Timeout", 0.26, "Timeout", "Timeout", 0, 188.59, "Timeout", 0.84, 167.65, "Timeout", "Timeout", "Timeout", 0.35, 0.39, 1.79, 1.25, 0.83]
eb_ctuf=[2.75, "Timeout", "Timeout", 39.98, 149.54, "Timeout", 0.56, 1.94, "Timeout", 0.92, "Timeout", 140.33, "Timeout", 1.1, 1.07, "Timeout", 273.23, "Timeout", "Timeout", 1.1, 1.27, 1.6, 2.58, "Timeout", 0.31, "Timeout", "Timeout", 0, 193.02, "Timeout", 0.89, 193.01, "Timeout", "Timeout", "Timeout", 0.43, 0.4, 2.24, 1.4, 0.87]
plot_sequences(eb_serial, eb_ctf,eb_ctuf, "lin_eb.png","EB",550)

