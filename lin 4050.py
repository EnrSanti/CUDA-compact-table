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


eb_serial_5=['Timeout', 'Timeout', 'Timeout', 'Timeout', 562.93, 'Timeout', 'Timeout', 208.07, 309.33, 333.21, 'Timeout', 385.05, 556.86, 'Timeout', 'Timeout', 851.03, 'Timeout', 'Timeout', 380.78, 565.19, 17.66, 'Timeout', 'Timeout', 'Timeout', 'Timeout', 16.64, 810.37, 235.85, 'Timeout', 'Timeout', 'Timeout', 25.16, 'Timeout', 'Timeout', 124.46, 'Timeout', 716.89, 'Timeout', 15.58, 480.36]
eb_ctf_5=[464.84, 'Timeout', 'Timeout', 'Timeout', 274.29, 'Timeout', 'Timeout', 45.16, 116.02, 88.63, 'Timeout', 248.52, 340.94, 'Timeout', 'Timeout', 463.22, 'Timeout', 'Timeout', 260.52, 331.85, 6.1, 678.88, 'Timeout', 'Timeout', 472.67, 5.74, 474.39, 50.9, 'Timeout', 'Timeout', 576.47, 8.69, 645.92, 'Timeout', 40.95, 558.71, 355.14, 'Timeout', 5.8, 288.88]
eb_ctuf_5=[568.96, 'Timeout', 'Timeout', 'Timeout', 329.93, 'Timeout', 'Timeout', 22.13, 139.83, 88.44, 'Timeout', 296.01, 419.14, 'Timeout', 'Timeout', 691.23, 'Timeout', 'Timeout', 314.32, 412.69, 9.78, 'Timeout', 'Timeout', 'Timeout', 668.18, 8.06, 628.44, 42.81, 'Timeout', 'Timeout', 737.03, 11.15, 868.07, 'Timeout', 20.75, 740.11, 439.13, 'Timeout', 10.12, 349.37]
b_serial_6=[468.91, 156.37, 'Timeout', 105.21, 237.75, 'Timeout', 189.56, 40.0, 51.81, 182.86, 'Timeout', 229.95, 256.94, 196.16, 272.89, 'Timeout', 499.76, 'Timeout', 'Timeout', 413.94, 280.29, 445.64, 107.16, 'Timeout', 45.88, 'Timeout', 'Timeout', 375.37, 440.91, 'Timeout', 227.71, 260.41, 'Timeout', 214.74, 'Timeout', 479.38, 533.73, 284.79, 489.02, 293.87]
b_ctf_6=[222.98, 82.68, 'Timeout', 43.85, 136.47, 'Timeout', 102.5, 9.89, 23.77, 124.01, 'Timeout', 132.99, 171.25, 130.36, 151.08, 'Timeout', 230.08, 'Timeout', 'Timeout', 223.63, 152.18, 205.67, 84.02, 'Timeout', 19.81, 'Timeout', 'Timeout', 152.15, 188.59, 'Timeout', 112.17, 167.65, 'Timeout', 119.04, 'Timeout', 206.67, 242.74, 172.46, 197.36, 165.78]
b_ctuf_6=[205.71, 77.08, 'Timeout', 39.98, 149.54, 'Timeout', 104.43, 6.241, 21.75, 139.5, 'Timeout', 140.33, 225.99, 136.68, 156.06, 'Timeout', 273.23, 'Timeout', 'Timeout', 241.41, 171.83, 237.1, 85.84, 'Timeout', 15.65, 'Timeout', 'Timeout', 147.17, 193.02, 'Timeout', 104.17, 193.01, 'Timeout', 128.65, 'Timeout', 217.74, 235.8, 203.18, 214.4, 236.99]

plot_sequences(eb_serial_5,  eb_ctf_5, eb_ctuf_5, "lin_eb.png","EB",950)
plot_sequences(b_serial_6, b_ctf_6,b_ctuf_6, "lin_b.png","B",550)
