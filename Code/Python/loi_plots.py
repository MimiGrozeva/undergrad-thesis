import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Get the valid columns
# df = pd.read_csv("ms_loi_for_plot.csv")
# filtered_df = df[~df["MS"].isna() & (df["MS"] != 0)]
# filtered_df.to_csv("loi_filtered.csv", index=False)

# def plot_vertical(x, y):
#     """
#     Plots a vertical line plot given x and y coordinates.
    
#     Parameters:
#     - x: A list or array of x coordinates (dependent variable).
#     - y: A list or array of y coordinates (independent variable).
#     """

    
#     # Plotting
#     fig = plt.figure()
#     ax = fig.add_subplot(131)

#     ax.plot(y, x, color="black",)
#     ax.invert_yaxis()
#     ax.set_ylabel('depth')
#     ax.set_xlabel('ms')
#     plt.show()


MS_df = pd.read_csv("loi_MS.csv") 

loi_df = pd.read_csv("loi.csv")




# Plotting
fig = plt.figure()

MS_ax = fig.add_subplot(151)
MS_ax.plot(MS_df["MS"].to_numpy(), MS_df["Depth"].to_numpy(), color="black",)
MS_ax.invert_yaxis()
MS_ax.set_ylabel('Depth (cm)')
MS_ax.set_xlabel('MS (1e-6)')
MS_ax.set_ylim(472,300)
MS_ax.xaxis.tick_top()
MS_ax.xaxis.set_label_position('top') 

loi_550 = fig.add_subplot(152)
loi_550.plot(loi_df["LOI 500"].to_numpy(), loi_df["Depth"].to_numpy(), color="black",)
loi_550.invert_yaxis()
loi_550.set_xlabel('Organic Carbon (550' + u"\u00B0" + ' LOI wt. %)')
loi_550.set_ylim(472,300)
loi_550.xaxis.tick_top()
loi_550.xaxis.set_label_position('top') 


loi_1000 = fig.add_subplot(153)
loi_1000.plot(loi_df["LOI 1000"].to_numpy(), loi_df["Depth"].to_numpy(), color="black",)
loi_1000.invert_yaxis()
loi_1000.set_xlabel('Inorganic Carbon (1000' + u"\u00B0" + ' LOI wt. %)')
loi_1000.set_ylim(472,300)
loi_1000.xaxis.tick_top()
loi_1000.xaxis.set_label_position('top') 

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.5, hspace=0.2)  # Adjust these values as needed



plt.show()
