
import psyplot.project as psy
import pandas as pd
from psy_strat.stratplot import stratplot
import matplotlib.pyplot as plt  # Import Matplotlib

import pandas as pd

def create_pollen_data_csv(site_csv_path, pollen_csv_path):
    """
    Create a pollen data CSV file from a site CSV file in the shape of berry_pond.csv.

    Args:
        site_csv_path (str): Path to the site CSV file.
        pollen_csv_path (str): Path to save the generated pollen data CSV file.

    Returns:
        None
    """
    # Load the site CSV file
    site_df = pd.read_csv(site_csv_path)
    
    # Reshape the DataFrame
    # Pivoting the DataFrame to have species as columns and their corresponding values
    # 'age' will be the index
    site_pivot_df = site_df[['depth', 'species', 'value']]
    site_pivot_df = site_pivot_df.pivot_table(index='depth', columns='species', values='value', aggfunc='sum')
    site_pivot_df.reset_index(inplace=True)
    
    # Export the reshaped DataFrame to a CSV file
    site_pivot_df.to_csv(pollen_csv_path, index=False)
    print(f"Pollen data CSV file '{pollen_csv_path}' has been created.")

# Example usage:
# Specify the paths for your site CSV and where you want to save the pollen data CSV
create_pollen_data_csv("C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Ponds\\berry_pond.csv", 'C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Ponds\\reshaped_berry_pond_data.csv')


df = pd.read_csv("C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Ponds\\reshaped_berry_pond_data.csv", index_col='depth')
df.head()


sp, groupers = stratplot(
    df,
    thresh=1.0,
    trunc_height=0.1,
    use_bars=['Herbs'],
    all_in_one=['Temperature'],
    summed=['Trees and shrubs', 'Herbs'],
    widths={'Temperature': 0.1, 'Pollen': 0.9},
    percentages=['Pollen'],
    calculate_percentages=True,
)





# -- psyplot update
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
red = '#d62728'

# change the color of trees and shrubs to green
sp(group='Trees and shrubs').update(color=[green])


# mark small taxon occurences below 1% with a +
sp(maingroup='Pollen').update(occurences=1.0)

# change the color of JJA temperature to red, shorten legend labels and
# change the x- and y-label
sp(group='Temperature').update(color=[blue, red], legendlabels=['DJF', 'JJA'],
                               ylabel='Age BP [years]', xlabel='$^\circ$C',
                               legend={'loc': 'lower left'})

# change the color of the summed trees and shrubs to green and put the legend
# on the bottom
sp(group='Summed').update(color=[green, orange], legend={'loc': 'lower left'})

# Display the plot (optional)
plt.show()

