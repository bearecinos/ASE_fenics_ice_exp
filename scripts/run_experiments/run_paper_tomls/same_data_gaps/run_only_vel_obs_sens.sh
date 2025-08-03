#!/bin/bash

# Get number of CPUs and shift the arguments
NUM_CPUS=$1
shift

# Assign CPU cores dynamically (you can adjust these manually if needed)
CPU1=0
CPU2=1
CPU3=2
CPU4=3

# Create and define your run_input directory for the workflow run
input_run_inv=$INPUT_DIR/input_run_workflow_test
if [ ! -d $input_run_inv ]
then
  echo "Creating run input directory $input_run_inv"
  mkdir $input_run_inv
else
  echo "Directory is $input_run_inv already exist"
fi

# Create output directory for inversion output
export run_inv_output_dir=$OUTPUT_DIR/09_same_data_gaps
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

path_logs=$RUN_CONFIG_DIR/run_paper_tomls/same_data_gaps
echo "Logs will be store here:" $path_logs

# Update TOML paths
for toml_path in "$@"; do
  toml set --toml-path "$toml_path" io.input_dir "$input_run_inv"
  toml set --toml-path "$toml_path" io.output_dir "$run_inv_output_dir/output"
  toml set --toml-path "$toml_path" io.diagnostics_dir "$run_inv_output_dir/diagnostics"
done

echo $(date -u) "Run vel obs sens 1 started" | mail -s "run vel obs sens started" beatriz.recinos@ed.ac.uk
taskset -c $CPU1 python "$FENICS_ICE_BASE_DIR/runs/run_obs_sens_prop.py" "$1" |& tee "$path_logs/log_vel_obs_sens_1.txt" &
PID1=$!

echo $(date -u) "Run vel obs sens 2 started" | mail -s "run vel obs sens started" beatriz.recinos@ed.ac.uk
taskset -c $CPU2 python "$FENICS_ICE_BASE_DIR/runs/run_obs_sens_prop.py" "$2" |& tee "$path_logs/log_vel_obs_sens_2.txt" &
PID2=$!

echo $(date -u) "Run vel obs sens 3 started" | mail -s "run vel obs sens started" beatriz.recinos@ed.ac.uk
taskset -c $CPU3 python "$FENICS_ICE_BASE_DIR/runs/run_obs_sens_prop.py" "$3" |& tee "$path_logs/log_vel_obs_sens_3.txt" &
PID3=$!

echo $(date -u) "Run vel obs sens 4 started" | mail -s "run vel obs sens started" beatriz.recinos@ed.ac.uk
taskset -c $CPU4 python "$FENICS_ICE_BASE_DIR/runs/run_obs_sens_prop.py" "$4" |& tee "$path_logs/log_vel_obs_sens_4.txt" &
PID4=$!

# Wait for all jobs to finish
wait $PID1 $PID2 $PID3 $PID4

# Send completion emails
OUT1=$(tail "$path_logs/log_vel_obs_sens_1.txt")
echo "$OUT1" | mail -s "run vel obs sens 1 finished" beatriz.recinos@ed.ac.uk

OUT2=$(tail "$path_logs/log_vel_obs_sens_2.txt")
echo "$OUT2" | mail -s "run vel obs sens 2 finished" beatriz.recinos@ed.ac.uk

OUT3=$(tail "$path_logs/log_vel_obs_sens_3.txt")
echo "$OUT3" | mail -s "run vel obs sens 3 finished" beatriz.recinos@ed.ac.uk

OUT4=$(tail "$path_logs/log_vel_obs_sens_4.txt")
echo "$OUT4" | mail -s "run vel obs sens 4 finished" beatriz.recinos@ed.ac.uk
