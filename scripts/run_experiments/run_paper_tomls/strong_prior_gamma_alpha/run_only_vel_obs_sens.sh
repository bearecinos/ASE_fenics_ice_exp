#!/bin/bash

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
export run_inv_output_dir=$OUTPUT_DIR/07_gamma_alpha_10000
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

path_logs=$RUN_CONFIG_DIR/run_paper_tomls/strong_prior_gamma_alpha
echo "Logs will be store here:" $path_logs

toml set --toml-path $1 io.input_dir "$input_run_inv"
toml set --toml-path $1 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $1 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

echo $(date -u) "Run vel obs sens started" | mail -s "run vel obs sens started" beatriz.recinos@ed.ac.uk
python $FENICS_ICE_BASE_DIR/runs/run_obs_sens_prop.py $1 |& tee $path_logs/log_vel_obs_sens.txt
OUT=$(tail "$path_logs/log_vel_obs_sens.txt")
echo $OUT | mail -s "run vel obs sens finished" beatriz.recinos@ed.ac.uk

