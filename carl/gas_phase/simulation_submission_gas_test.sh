#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --array=1-2%2
#SBATCH -G 1

# Print a message indicating the start of the script
echo "Starting SLURM script..."

# Extract Md_Experiment_uuid from the first line of gas_phase.txt
Md_Experiment_uuid=$(awk "NR==1 {print}" gas_phase.txt | cut -d' ' -f2)
echo "Md_Experiment_uuid: $Md_Experiment_uuid"

# Define the number of calculations to run in each loop
STEP=1

# Define the maximum index of calculations
MAXINDEX=$(wc -l < gas_phase.txt)

# Calculate the starting and ending indices for this job
START_INDEX=$((($SLURM_ARRAY_TASK_ID - 1) * $STEP + 1))
END_INDEX=$((START_INDEX + STEP - 1))

# Check if the END_INDEX exceeds MAXINDEX and adjust it if necessary
if [ $END_INDEX -gt $MAXINDEX ]; then
    END_INDEX=$MAXINDEX
fi

# Run calculations in a loop
echo "Running calculations..."
for ((i=$START_INDEX; i<=$END_INDEX; i++)); do
    confid=$(awk "NR==$i {print}" gas_phase.txt | cut -d' ' -f1)
    if [ -f mols_3d/$confid.mol ]; then
        python_command="python mdfp_liquid.py $confid $Md_Experiment_uuid"
        echo "Running Python command: $python_command"
        # Run the Python script for each confid and the extracted Md_Experiment_uuid
        echo "Running Python script for confid $confid..."
        python mdfp_liquid.py $confid $Md_Experiment_uuid
    else
        echo "File mols_3d/$confid.mol does not exist, skipping..."
    fi
done

# Print a message indicating the end of the script
echo "SLURM script completed."

