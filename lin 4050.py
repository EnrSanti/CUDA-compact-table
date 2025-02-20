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


eb_serial_5=['Timeout', 'Timeout', 'Timeout', 'Timeout', 562.93, 'Timeout', 'Timeout', 'Timeout', 309.33, 'Timeout', 'Timeout', 385.05, 12.57, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 565.19, 17.66, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 11.08, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 11.15, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 716.89, 'Timeout', 15.58]
eb_ctf_5=['Timeout', 'Timeout', 'Timeout', 'Timeout', 274.29, 'Timeout', 'Timeout', 'Timeout', 116.02, 'Timeout', 'Timeout', 248.52, 4.57, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 331.85, 6.1, 678.88, 'Timeout', 'Timeout', 472.67, 4.48, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 4.57, 645.92, 'Timeout', 'Timeout', 558.71, 355.14, 'Timeout', 5.8]
eb_ctuf_5=['Timeout', 'Timeout', 'Timeout', 'Timeout', 329.93, 'Timeout', 'Timeout', 'Timeout', 139.83, 'Timeout', 'Timeout', 296.01, 6.61, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 412.69, 9.78, 'Timeout', 'Timeout', 'Timeout', 668.18, 6.61, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 7.17, 868.07, 'Timeout', 'Timeout', 740.11, 439.13, 'Timeout', 10.12]
b_serial_6=['Timeout', 'Timeout', 'Timeout', 105.21, 237.75, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 229.95, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 499.76, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 440.91, 'Timeout', 'Timeout', 260.41, 'Timeout', 'Timeout', 'Timeout']
b_ctf_6=['Timeout', 'Timeout', 'Timeout', 43.85, 136.47, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 132.99, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 230.08, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 188.59, 'Timeout', 'Timeout', 167.65, 'Timeout', 'Timeout', 'Timeout']
b_ctuf_6=['Timeout', 'Timeout', 'Timeout', 39.98, 149.54, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 140.33, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 273.23, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 'Timeout', 193.02, 'Timeout', 'Timeout', 193.01, 'Timeout', 'Timeout', 'Timeout']

plot_sequences(eb_serial_5,  eb_ctf_5, eb_ctuf_5, "lin_eb.png","EB",950)
plot_sequences(b_serial_6, b_ctf_6,b_ctuf_6, "lin_b.png","B",550)
