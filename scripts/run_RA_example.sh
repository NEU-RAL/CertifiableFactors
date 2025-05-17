#!/bin/bash

# Declare array with arguments formatted correctly
args_list=(
    "2 2 ais2klinik ais2kliniktestp4"
    "2 2 city10000 city10000testp4"
    "2 2 CSAIL CSAILtestp4"
    "2 2 intel inteltestp4"
    "2 2 kitti_00 kitti_00testp4"
    "2 2 kitti_02 kitti_02testp4"
    "2 2 kitti_05 kitti_05testp4"
    "2 2 manhattan manhattantestp4"
    "2 2 MIT MITtestp4"
    "3 3 cubicle cubicletestp5"
    "3 3 grid3D grid3Dtest"
    "3 3 parking-garage parkinggaragetestp5"
    "3 3 rim rimtestp5"
    "3 3 sphere2500 sphere2500testp5"
    "3 3 sphere_bignoise_vertex3 sphere_bignoise_vertex3testp5"
    "3 3 smallGrid3D smallGrid3Dtestp5"
    "3 3 torus3D torus3Dtestp5"
)

# Correct variable assignment without spaces
SESYNC_PATH="./cmake-build-release/Preconditioners/bin/Certifiable_RA_example"

# Loop through the array and run the command for each set of arguments
for args in "${args_list[@]}"; do
    # Split each string into its respective parts
    read -r d p dataset output <<< "$args"

    # Echo the arguments for debugging purposes
    echo "Running with arguments: d=$d, p=$p, dataset=$dataset, output=$output"

    # Call the program with the correct arguments
    $SESYNC_PATH "$d" "$p" "$dataset" "$output"
done
