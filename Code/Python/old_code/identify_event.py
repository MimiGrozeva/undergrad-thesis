import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def sigmoid(x, x0, k):
    """Sigmoid function used to model the transition event."""
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return y

def detect_event(pollen_percentages):
    """
    Detects a significant event in a series of pollen percentages.
    
    Parameters:
    - pollen_percentages: 1D array of pollen percentages for a given species.
    
    Returns:
    - event_detected: Whether an event was detected.
    - event_location: The index in the array where the event is most pronounced.
    """
    # Normalize the data to range between 0 and 1 for fitting
    normalized_data = (pollen_percentages - np.min(pollen_percentages)) / (np.max(pollen_percentages) - np.min(pollen_percentages))
    x_data = np.arange(len(normalized_data))
    
    # Attempt to fit a sigmoid function to the data
    try:
        popt, _ = curve_fit(sigmoid, x_data, normalized_data, method='dogbox', bounds=([min(x_data), 0.01], [max(x_data), 1]))
        event_location = popt[0]
        event_detected = True
    except:
        event_detected = False
        event_location = None
    
    return event_detected, event_location, popt

df = pd.read_csv("C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Processed_Pollens\\knob_hill_processed.csv")

numerical_df = df.iloc[3:].copy()

event_detected, event_location, popt= detect_event(numerical_df['Tsuga'].astype(float).tolist())

print(event_detected)
print(event_location)

# Plotting
if event_detected:
    plt.figure(figsize=(10, 6))
    plt.plot(numerical_df['Tsuga'].astype(float).tolist(), label='Tsuga Pollen Percentages')
    plt.plot(np.arange(len(numerical_df)), sigmoid(np.arange(len(numerical_df)), *popt), color='red', label='Fitted Sigmoid')
    plt.axvline(x=event_location, color='green', linestyle='--', label='Event Location')
    plt.title('Tsuga Pollen Percentages and Detected Event')
    plt.xlabel('Sample Index')
    plt.ylabel('Pollen Percentage')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No significant event was detected.")