import numpy as np
import os

import matplotlib.pyplot as plt

dir_name = "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Code\\Python\\estimated_depths"
filenames = [
    "ware_ponds_processed_depths.txt",
    "berry_pond_processed_depths.txt", 
    "guilder_pond_processed_depths.txt",
    "knob_hill_processed_depths.txt", 
    "little_royalston_processed_depths.txt", 
    "little_willey_processed_depths.txt", 
    "twin_ponds_processed_depths.txt"
]

ware_depths = [460.0, 480.0, 500.0, 525.0, 550.0, 575.0, 600.0]
berry_depths = [302.0, 306.0, 310.0, 314.0, 318.0, 322.0, 326.0, 330.0, 334.0, 338.0, 342.0, 346.0, 350.0, 354.0, 358.0, 362.0, 366.0, 370.0, 374.0, 378.0, 390.0, 410.0, 430.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0, 505.0, 510.0, 520.0, 530.0, 550.0, 560.0, 565.0, 570.0, 575.0, 580.0, 590.0, 610.0, 630.0, 650.0, 660.0, 665.0, 670.0, 675.0, 680.0, 690.0, 710.0, 730.0]
guilder_depths = [88.0, 96.0, 104.0, 112.0, 120.0, 130.0, 140.0, 150.0, 154.0, 158.0, 160.0, 162.0, 164.0, 166.0, 168.0, 170.0, 172.0]
knob_depths = [343.0, 353.0, 363.0, 373.0, 383.0, 393.0, 403.0, 413.0, 423.0, 433.0, 443.0, 453.0, 463.0, 473.0, 477.0, 483.0, 487.0, 493.0, 497.0, 503.0]
royalston_depths = [264.0, 272.0, 280.0, 288.0, 296.0, 300.0, 304.0, 308.0, 312.0, 320.0, 328.0, 332.0, 336.0, 352.0, 368.0, 384.0, 392.0, 396.0, 400.0, 408.0, 416.0, 432.0, 448.0, 456.0, 460.0, 464.0, 468.0, 472.0, 480.0, 496.0, 512.0, 528.0]
willey_depths = [145.5, 156.0, 177.0, 198.0, 219.0, 230.0, 240.0, 245.0, 250.0, 255.0, 261.0, 266.0, 271.5, 277.0, 282.0, 287.0, 292.5, 298.0, 303.0, 308.0, 313.0, 318.0, 324.0, 334.5, 345.0]
twin_depths = [284.5, 289.5, 294.5, 304.5, 309.5, 314.5, 322.5, 327.5, 332.5, 337.5, 342.5, 352.5, 362.5, 372.5, 382.5, 392.5, 402.5, 411.5, 419.5, 424.5, 429.5, 439.5]

depths = [ware_depths, berry_depths, guilder_depths, knob_depths, royalston_depths, willey_depths, twin_depths]

for i in range(len(filenames)):
    f = filenames[i]
    possible_depths = depths[i]

    data_array = np.loadtxt(os.path.join(dir_name,f))

    # Load data from file
    data_array = np.loadtxt(os.path.join(dir_name, f))
    
    # Determine the range of unique values in data_array
    unique_values = np.unique(data_array)
    min_val = unique_values.min()
    max_val = unique_values.max()
    
    # Limit the bins to be within +/- 1 of the range of unique values
    filtered_depths = [d for d in possible_depths if (min_val - 1) <= d <= (max_val + 1)]
    
    # Extend the bin range slightly to include all data
    if filtered_depths[0] > min_val - 1:
        filtered_depths.insert(0, min_val - 1)
    if filtered_depths[-1] < max_val + 1:
        filtered_depths.append(max_val + 1)
    
    # Create a histogram with refined bin edges
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(data_array, bins=filtered_depths, edgecolor='black')

    # Calculate tick positions as the midpoint between consecutive bin edges
    tick_positions = bins[:-1] + np.diff(bins) / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{val:.1f}" for val in tick_positions])

    # Set labels and title
    ax.set_xlabel('Depth Values')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram for {f}')

    # Display the histogram
    plt.show()