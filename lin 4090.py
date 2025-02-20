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



b_serial_5=['Timeout', 'Timeout', 'Timeout', 'Timeout', 429.71, 'Timeout', 'Timeout', 'Timeout', 226.79, 'Timeout', 'Timeout', 300.15, 9.7, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 427.8, 13.33, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 8.71, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 8.73, 'Timeout', 'Timeout', 'Timeout', 837.51, 545.67, 'Timeout', 12.05]
b_ctf_5=['Timeout', 'Timeout', 'Timeout', 'Timeout', 222.921, 'Timeout', 'Timeout', 'Timeout', 104.93, 'Timeout', 'Timeout', 216.25, 4.25, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 278.33, 5.62, 653.7, 'Timeout', 'Timeout', 470.87, 4.21, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 4.3, 616.32, 'Timeout', 'Timeout', 449.05, 302.91, 'Timeout', 5.74]
b_ctuf_5=['Timeout', 'Timeout', 'Timeout', 'Timeout', 205.69, 'Timeout', 'Timeout', 'Timeout', 86.0, 'Timeout', 'Timeout', 197.3, 3.76, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 257.14, 5.13, 570.65, 'Timeout', 'Timeout', 418.47, 3.76, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 3.96, 519.7, 'Timeout', 'Timeout', 427.53, 270.32, 'Timeout', 5.22]
eb_serial_6=['Timeout', 'Timeout', 'Timeout', 80.55, 181.43, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 181.32, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 385.71, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 342.51, 'Timeout', 'Timeout', 204.92, 'Timeout', 'Timeout', 'Timeout']
eb_ctf_6=['Timeout', 'Timeout', 'Timeout', 34.34, 125.7, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 118.224, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 209.13, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 161.94, 'Timeout', 'Timeout', 148.49, 'Timeout', 'Timeout', 'Timeout']
eb_ctuf_6=['Timeout', 'Timeout', 'Timeout', 28.23, 111.39, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 103.57, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 185.42, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 142.27, 'Timeout', 'Timeout', 132.83, 'Timeout', 'Timeout', 'Timeout']


plot_sequences(b_serial_5,  b_ctf_5, b_ctuf_5, "lin_b_4090.png","B",950)
plot_sequences(eb_serial_6, eb_ctf_6,eb_ctuf_6, "lin_eb_4090.png","EB",550)

