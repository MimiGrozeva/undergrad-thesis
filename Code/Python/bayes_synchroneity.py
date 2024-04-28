# load libraries
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import gaussian_kde
from scipy.integrate import quad

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

import os
import itertools

def generate_samples(min_val, max_val, median, mean, num_samples=10000):
    """
    Generates num_samples samples for a distribution given its mean, min, max, and median.
    This function assumes a skewed distribution if mean and median are not equal and
    adjusts the distribution shape accordingly.
    
    Parameters:
    - mean: The mean of the distribution.
    - min_val: The minimum value of the distribution.
    - max_val: The maximum value of the distribution.
    - median: The median of the distribution.
    - num_samples: The number of samples to generate (default 10,000).
    
    Returns:
    - samples: An array of num_samples samples from the distribution.
    """
    # Estimate skewness based on the mean, median, and mode.
    # This is a simplistic approach for demonstration purposes.
    skewness = 0
    if mean > median:
        # Positively skewed distribution
        skewness = (mean - median) / (max_val - mean)
    elif mean < median:
        # Negatively skewed distribution
        skewness = (median - mean) / (mean - min_val)
    
    # Generate samples using a beta distribution as it can be skewed and bounded.
    # Adjust the alpha and beta parameters based on estimated skewness.
    alpha, beta = 2, 2  # Start with a symmetric shape
    if skewness > 0:
        alpha *= 1 + skewness
    elif skewness < 0:
        beta *= 1 - skewness
    
    # Generate samples from a beta distribution and scale to [min_val, max_val].
    raw_samples = np.random.beta(alpha, beta, num_samples)
    samples = raw_samples * (max_val - min_val) + min_val
    
    return samples

def generate_confidence_intervals(samples1, samples2, confidence_level=.950):
    """
    Calculate the 95% confidence intervals for two sets of samples and determine
    if they overlap, to assess synchroneity.
    
    Parameters:
    - samples1: An array of samples from the first site.
    - samples2: An array of samples from the second site.
    - confidence_level: The confidence level for the interval (use 95%).
    
    Returns:
    - ci1: The confidence interval for the first set of samples.
    - ci2: The confidence interval for the second set of samples.
    - overlap: Boolean indicating if the intervals overlap.
    """
    # Calculate the confidence intervals for each set of samples
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    
    ci1 = np.percentile(samples1, [lower_percentile, upper_percentile])
    ci2 = np.percentile(samples2, [lower_percentile, upper_percentile])
    
    # Determine if the intervals overlap
    overlap = not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
    
    return ci1, ci2, overlap

def generate_single_confidence_interval(samples1, confidence_level=.950):
    """
    Calculate the 95% confidence intervals for two sets of samples and determine
    if they overlap, to assess synchroneity.
    
    Parameters:
    - samples1: An array of samples from the first site.
    - samples2: An array of samples from the second site.
    - confidence_level: The confidence level for the interval (default is 97.5%).
    
    Returns:
    - ci1: The confidence interval for the first set of samples.
    - ci2: The confidence interval for the second set of samples.
    - overlap: Boolean indicating if the intervals overlap.
    """
    # Calculate the confidence intervals for each set of samples
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    
    ci = np.percentile(samples1, [lower_percentile, upper_percentile])

    return ci 

def fetch_age_distribution(event_depth, pond_file):
    """
    Fetches the age distribution (min_age, max_age, median_age, mean_age) for the specified event_depth from the DataFrame.
    
    Parameters:
    - event_depth: The exact depth for which we want to find the age distribution.
    - pond_file: The path to the CSV file containing the site data.
    
    Returns:
    - age_distribution: A dictionary containing the min_age, max_age, median_age, and mean_age for the specified depth.
    """
    df = pd.read_csv(pond_file)

    # Find the row that exactly matches the event_depth
    row = df[df['Depth'] == event_depth].iloc[0]
    
    # Extract the age distribution
    age_distribution = {
        'min_val': row['min_age'],
        'max_val': row['max_age'],
        'median': row['median_age'],
        'mean': row['mean_age']
    }

    return age_distribution

def perc_of_overlap(samples_A, samples_B):
    
    min_range = min(np.min(samples_A), np.min(samples_B))
    max_range = max(np.max(samples_A), np.max(samples_B))

    # Estimate the density functions using KDE
    kde_A = gaussian_kde(samples_A)
    kde_B = gaussian_kde(samples_B)

    def min_density(x):
        return min(kde_A(x), kde_B(x))

    overlap_area, _ = quad(min_density, min_range, max_range)

    return overlap_area


def determine_sync(pond_file1, pond_file2, pond_name1, pond_name2, event_depth1, event_depth2):
    # Get the distributions
    distribution1 = fetch_age_distribution(event_depth1, pond_file1)
    distribution2 = fetch_age_distribution(event_depth2, pond_file2)

    # Generate samples over those distibutions
    samples1 = generate_samples(**distribution1)
    samples2 = generate_samples(**distribution2)

    # Calculate overlap with confidence level of 95%
    ci1, ci2, overlap = generate_confidence_intervals(samples1, samples2)

    # P(A&B)
    perc_overlap = perc_of_overlap(samples1, samples2)

    # P(A | B) & P(B | A) = ( P(A&B) / P(A) ) * (P(A&B) / P(B)).
    a_given_b = perc_overlap / quad(gaussian_kde(samples1), np.min(samples1), np.max(samples1))[0]
    b_given_a = perc_overlap / quad(gaussian_kde(samples2), np.min(samples2), np.max(samples2))[0]

    prob_co_occur = a_given_b * b_given_a


    # Display results
    print(f"95% Confidence Interval for {name1}: {ci1}")
    print(f"95% Confidence Interval for {name2}: {ci2}")
    print(f"Percentage of overlap: {perc_overlap}")
    print(f"Probabilty Co-Occur: {round(prob_co_occur*100, 5) }%")
    print(f"Do the intervals overlap? {'Yes' if overlap else 'No'}")

    # Determine synchrony or diachroneity
    if overlap:
        print("The events are synchronous - they occurred approximately at the same time across the different locations.")
    else:
        print("The events are diachronous - they occurred at different times in different places.")

    return prob_co_occur

def plot_samples_with_confidence_intervals(sample_sets, confidence_intervals, labels, colors):
    """
    Plots density plots for samples from different distributions and colors the area under the curve
    only between the confidence intervals, using specified colors for each plot and interval.
    
    Parameters:
    - sample_sets: A list of arrays, each containing samples from a distribution.
    - confidence_intervals: A list of tuples, each representing the confidence interval for the corresponding set of samples.
    - labels: A list of strings, labels for each sample set.
    - colors: A list of color codes (strings) for the density curve and the interval area.
    """
    plt.figure(figsize=(12, 8))
    
    for samples, ci, label, color in zip(sample_sets, confidence_intervals, labels, colors):
        # Generate the density plot data
        density = sns.kdeplot(samples, lw=2, color=color, alpha=1).get_lines()[-1].get_data()
        x, y = density  # x and y values of the density plot
        
        # Find indices where x is within the confidence interval
        ci_mask = (x >= ci[0]) & (x <= ci[1])
        
        # Fill the area under the plot within the confidence interval
        plt.fill_between(x, y, where=ci_mask, color=color, alpha=0.3, label=f"{label}")
    
    
    
    plt.xlabel('Age (cal yr BP)')
    plt.ylabel('Density')
    plt.title('Age Distributions of $\it{T.}$ $\it{canadensis}$ Decline')
    plt.legend()
    plt.xlim(4500, 6500)  # Set the x-axis to range from 3000 to 7000
    plt.savefig("C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Figures\\TsugaCIs.pdf")

# Plotting utility
def gradient_fill(x, y, fill_color=None, ax=None, label=None, interval = None, **kwargs):
    if ax is None:
        ax = plt.gca()
    
    # Convert x and y to numpy arrays if they are not already
    x = np.array(x)
    y = np.array(y)

    line, = ax.plot(x, y, color = fill_color, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    
    # Add vertical dotted lines at the specified intervals
    if interval is not None and len(interval) == 2:
        for xval in interval:
            ax.axvline(x=xval, color='k', linestyle=':', linewidth=1.5)

    # Add the label as a custom legend entry without a line
    if label is not None:
        legend_line = Line2D([0], [0], label=label, color='none', marker='None')
        ax.legend(handles=[legend_line], handlelength=0, handletextpad=0)

    ax.autoscale(True)
    return line, im

# More plot functions
def plot_tsuga(tsuga_counts, ages, labels, colors, intervals, mean_ages):
    n = len(tsuga_counts)
    fig, axs = plt.subplots(n, 1, figsize=(8, 2 * n))
    
    if n == 1:
        axs = [axs]

    for i, (counts, age) in enumerate(zip(tsuga_counts, ages)):
        gradient_fill(age, counts, fill_color=colors[i], ax=axs[i], label=labels[i], interval = intervals[i])

        if i < n - 1:
            # plt.xticks([mean_ages[i]])
            axs[i].xaxis.set_ticks([mean_ages[i]])
            axs[i].set_xlim([0, 12000])
            # axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        else:            
            plt.xticks(list(plt.xticks()[0]) + [mean_ages[i]])
            axs[i].set_xlabel('Age (cal yr BP)')
            axs[i].set_xlim([0, 12000])
        axs[i].set_ylabel('Pollen %')

        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(True)
        axs[i].spines['bottom'].set_visible(True)
    
    # plt.show()
    plt.tight_layout(pad=3.0)
    plt.savefig("C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Figures\\TsugaPercentages.pdf")





# Code to Run

# Parameters for synchroneity
base_input_url = "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Processed_Pollens\\"

in_names = [
    "knob_hill_processed.csv",
    "twin_ponds_processed.csv",
    "little_willey_processed.csv",
    "ware_ponds_processed.csv",
    "berry_pond_processed.csv",
    "little_royalston_processed.csv",
    "guilder_pond_processed.csv",
]
in_depths = [
    483, # knob hill,
    352.5, # twin
    250, # little willey
    575, # Ware Pond
    505, # berry
    332, # royalston
    162, # guilder pond
]
in_mean_ages = [
    5971, # knob hill
    5242, # twin
    5435, # little willey
    5340, # Ware Pond
    5244, # berry, 
    5204, # royalston
    4995, # guilder
]
colors = ['#115f9a', '#22a7f0','#76c68f','#c9e52f', '#9080ff', "#fcba03", "#e366e1"]

labels = [
    "Knob Hill",
    "Twin",
    "Little Willey",
    "Ware",
    "Berry",
    "Little Royalston",
    "Guilder",
]



# Generate all unique pairs of files and depths
pairs = list(itertools.combinations(zip(in_names, in_depths), 2))

# Create probability Matrix
names = list(set(labels[in_names.index(name)] for pair in pairs for name in (pair[0][0], pair[1][0])))
prob_matrix = pd.DataFrame(index=names, columns=names, dtype=float)


# Iterate over all pairs and call determine_sync for each
index = 0
for (name1, depth1), (name2, depth2) in pairs:
    pond_file1 = os.path.join(base_input_url, name1)
    pond_file2 = os.path.join(base_input_url, name2)
    prob_to_return = determine_sync(pond_file1, pond_file2, name1, name2, depth1, depth2)

    # Ensure that prob_to_return is numeric, and handle cases where it might not be
    try:
        # Convert to float, assuming the returned value should be numeric
        prob_to_return = float(prob_to_return)
    except ValueError:
        # If conversion fails, set it as NaN
        prob_to_return = np.nan


    # Storing prob_to_return in the DataFrame
    # Here, we use a MultiIndex of (name1, name2) to identify each pair uniquely
    name1_index = in_names.index(name1)
    name2_index = in_names.index(name2)
    prob_matrix.loc[labels[name1_index], labels[name2_index]] = prob_to_return
    prob_matrix.loc[labels[name2_index], labels[name1_index]] = prob_to_return
    prob_matrix.loc[labels[name1_index], labels[name1_index]] = np.nan


# Convert probability matrix to numeric
prob_matrix = prob_matrix.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

# Save Probability Matrix
prob_matrix.to_csv('prob_to_return_matrix.csv')

# Creating a heatmap from the matrix
plt.figure(figsize=(10, 8))
# Improve layout to prevent clipping of tick labels and labels
sns.heatmap(prob_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.tight_layout()
# plt.title('Heatmap of Probability to Return')
plt.show()
plt.savefig("C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Figures\\heatmap.pdf")


# Plots
samples = []
intervals = []
tsuga_counts = []
ages = []

for i in range(len(in_names)):
    # ci intervals
    pond_file = os.path.join(base_input_url, in_names[i])
    dist = fetch_age_distribution(in_depths[i], pond_file)
    s = generate_samples(**dist)
    samples.append(s)
    intervals.append(generate_single_confidence_interval(s))

    # tsuga
    df = pd.read_csv(pond_file)
    numerical_df = df.iloc[3:].copy()
    tsuga_counts.append(numerical_df['Tsuga'].astype(float).tolist())

    # ages
    ages.append(numerical_df['mean_age'].astype(float).tolist())

plot_samples_with_confidence_intervals(samples, intervals, labels, colors)
plot_tsuga(tsuga_counts, ages, labels, colors, intervals, in_mean_ages)



