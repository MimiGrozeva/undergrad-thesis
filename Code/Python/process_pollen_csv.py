# Maria Grozeva, 2024
# Code to process a raw neotoma csv for pollen plots;
# calculate percentages
#and append age-depth model data to those files

import pandas as pd


# age_file: filepath to ages generated by Bacon
# depths: python list of *integer* depths
def fit_bacon_age(age_file, depths):
    # load in ages
    bacon_df = pd.read_csv(age_file, delimiter="\t")

    bacon_df = bacon_df.loc[bacon_df["depth"].isin(depths)]
    bacon_df.reset_index(drop=True, inplace=True)

    return bacon_df

# function to reformat original files
def process_pollen_csv(
    input_file_path, output_file_path, bacon_file, is_new_format=False
):
    # Load the dataset
    df = pd.read_csv(input_file_path)

    # Names of columns to be removed, including "Author Submitted"
    if is_new_format:
        columns_to_remove_names = [
            "AnalysisUnitName",
            "Thickness",
            "Sample Name",
            "Sample ID",
            "Author generated",
            "units",
            "context",
        ]
    else:
        columns_to_remove_names = [
            "AnalysisUnitName",
            "Thickness",
            "Sample Name",
            "Sample ID",
            "Author submitted",
            "units",
            "context",
        ]
    # Remove columns based on 'name' field content
    df_cleaned = df[~df["name"].isin(columns_to_remove_names)]

    # Remove some unwanted data (spike, aquatics, spores, fragments)
    df_cleaned = df_cleaned[~df_cleaned["name"].str.contains("Spike", na=False)]
    df_cleaned = df_cleaned[~df_cleaned["group"].str.contains("AQVP", na=False)]
    df_cleaned = df_cleaned[~df_cleaned["element"].str.contains("spore", na=False)]
    df_cleaned = df_cleaned[~df_cleaned["element"].str.contains("fragment", na=False)]

    # Transpose the cleaned dataframe
    transposed_df = df_cleaned.T

    # Set the first row as the new column names
    transposed_df.columns = transposed_df.iloc[0]

    # Drop the first row
    transposed_df = transposed_df[1:]

    # Optionally, if you want to reset the index
    transposed_df.reset_index(drop=True, inplace=True)

    # Replace NaN values with 0
    transposed_df_filled = transposed_df.fillna(0)
    

    # Drop an empty row. Not sure why its here.
    processed_df = transposed_df_filled.drop(3)
    processed_df.reset_index(drop=True, inplace=True)  # reset index

    # Calculate percentages
    sum_exclude_cs = ["Depth"]
    sum_exclude_rows = [0, 1, 2]
    sum_df = processed_df.drop(columns=sum_exclude_cs).drop(index=sum_exclude_rows)
    sum_df = sum_df.astype(float)
    sum_df.iloc[:, :-1] = (sum_df.iloc[:, :-1].div(sum_df.sum(axis=1), axis=0)) * 100

    # Replace area of df with percentages
    processed_df.update(sum_df)

    # Get depths:
    depths = processed_df["Depth"].astype(float).tolist()[3:]

    # Add bacon ages
    bacon_ages = fit_bacon_age(bacon_file, depths)

    processed_df["Depth"].iloc[:3] = None

    # Add bacon values
    # First, calculate the "min_age" values without directly adding them to the DataFrame
    min_age_values = processed_df["Depth"].apply(
        lambda x: (
            None if x is None else
            bacon_ages.loc[bacon_ages["depth"] == (float(x)), "min"].values[0]
            if not bacon_ages.loc[bacon_ages["depth"] == (float(x)), "min"].empty
            else None
        )
    )
    # Now, insert "min_age" as the second column
    processed_df.insert(1, "min_age", min_age_values)

    # First, calculate the "max_age" values without directly adding them to the DataFrame
    max_age_values = processed_df["Depth"].apply(
        lambda x: (
            None if x is None else
            bacon_ages.loc[bacon_ages["depth"] == (float(x)), "max"].values[0]
            if not bacon_ages.loc[bacon_ages["depth"] == (float(x)), "max"].empty
            else None
        )
    )
    # Now, insert "max_age" as the third column
    processed_df.insert(2, "max_age", max_age_values)

    # First, calculate the "median_age" values without directly adding them to the DataFrame
    median_age_values = processed_df["Depth"].apply(
        lambda x: (
            None if x is None else
            bacon_ages.loc[bacon_ages["depth"] == (float(x)), "median"].values[0]
            if not bacon_ages.loc[bacon_ages["depth"] == (float(x)), "median"].empty
            else None
        )
    )
    # Now, insert "median_age" as the fourth column
    processed_df.insert(3, "median_age", median_age_values)

    # First, calculate the "mean" values without directly adding them to the DataFrame
    mean_age_values = processed_df["Depth"].apply(
        lambda x: (
            None if x is None else
            bacon_ages.loc[bacon_ages["depth"] == (float(x)), "mean"].values[0]
            if not bacon_ages.loc[bacon_ages["depth"] == (float(x)), "mean"].empty
            else None
        )
    )
    # Now, insert "mean_age" as the fifth column
    processed_df.insert(4, "mean_age", mean_age_values)

    # Save the modified dataframe to a new CSV file
    processed_df.to_csv(output_file_path, header=True, index=False)


# Testing
# input_file_path = "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Ponds\\test\\berry_pollen.csv"
# temp_file = "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Ponds\\test\\berry_pollen_processed.csv"
# bacon_file = "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Bacon_runs\\BerryPond\\BerryPond_120_ages.txt"
# process_pollen_csv(input_file_path, temp_file, bacon_file, False)

base_input_url = "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Ponds_New_Raw\\"
base_output_url = "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Processed_Pollens\\"

#names of my raw input files as downloaded from Neotoma
in_names = [
    "berry_pond_neotoma_raw.csv",
    "guilder_neotoma_raw.csv",
    "knob_hill_neotoma_raw.csv",
    "little_royalston_neotoma_raw.csv",
    "little_willey_neotoma_raw.csv",
    "twin_ponds_neotoma_raw.csv",
    "ware_neotoma_raw.csv",
    
]

# names of the output files
out_names = [
    "berry_pond_processed.csv",
    "guilder_pond_processed.csv",
    "knob_hill_processed.csv",
    "little_royalston_processed.csv",
    "little_willey_processed.csv",
    "twin_ponds_processed.csv",
    "ware_ponds_processed.csv",
]

# load files containing the age-depth data
age_names = [
    "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Bacon_runs\\BerryPond\\BerryPond_252_ages.txt",
    "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Bacon_runs\\GuilderPond\\GuilderPond_88_ages.txt",
    "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Bacon_runs\\KnobHillPond\\KnobHillPond_166_ages.txt",
    "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Bacon_runs\\LittleRoyalston\\LittleRoyalston_156_ages.txt",
    "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Bacon_runs\\LittleWilleyPond\\LittleWilleyPond_142_ages.txt",
    "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Bacon_runs\\TwinPonds\\TwinPonds_103_ages.txt",
    "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Bacon_runs\\WarePond\\WarePond_174_ages.txt",
]


for i in range(len(in_names)):
    is_new = False
    if in_names[i] == "twin_ponds_neotoma_raw.csv":
        is_new = True
    input_url = base_input_url + in_names[i]
    output_url = base_output_url + out_names[i]
    process_pollen_csv(input_url, output_url, age_names[i], is_new)

# Apply the function to the pollen data
# process_pollen_csv(input_file, temp_file)
