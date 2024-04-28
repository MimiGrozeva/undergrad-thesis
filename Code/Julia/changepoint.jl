# load necessary packages
# StatGeochem: changepoint
#Statistics, Plots, StatsBase are for plotting
using StatGeochem, Statistics, Plots, StatsBase

# Use Changepoint to determine depth of event
function get_best_depth(ds, cutoff_min, cutoff_max, name)
    # Subselect Tsuga using handpicked cutoffs
    tsuga = ds.Tsuga[cutoff_min:cutoff_max]

    # Run changepoint
    dist = changepoint(tsuga, 10_000_000, np=1)[100_000:end]

    # Convert distribution of indices to distribution of depths
    depthdist = (ds.Depth)[dist .+ cutoff_min .- 1]

    # Return the most common depth in the distribution
    return findmax(countmap(depthdist))[2]
end

# Parameters for our changepoint functions
filepath = "C:\\Users\\maria\\Desktop\\Research\\Quaternary\\Processed_Pollens"
filenames = [
    "ware_ponds_processed",
    "berry_pond_processed", 
    "guilder_pond_processed",
    "knob_hill_processed", 
    "little_royalston_processed", 
    "little_willey_processed", 
    "twin_ponds_processed"
]
# Cutoffs are converted from depths to indexes
cutoffs = [(42,48), (80, 130), (15,31), (75,94), (39,70), (31,55), (29, 50)]


for (i, f) in enumerate(filenames)
    print(f, "\n")

    # Read in pollen data
    ds = importdataset(joinpath(filepath, f*".csv"), ',', importas=:Tuple)

    # Subselect dates for changepoint
    min_depth = ds.Depth[cutoffs[i][1]]
    max_depth = ds.Depth[cutoffs[i][2]]
    
    # Choose optimum depth according to 50 discrete runs of changepoint,
    depths = []
    for j in 1:50
        append!(depths, get_best_depth(ds, cutoffs[i][1], cutoffs[i][2], f))
    end
    print("best depth: ", countmap(depths))
     
    
    # Store the outputs of the changepoint runs
    output_filename = joinpath(filepath, f * "_depths.txt")
    open(output_filename, "w") do file
        write(file, join(depths, "\n"))
    end

    print("\n\n")
end
