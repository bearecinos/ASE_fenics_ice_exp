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

toml set --toml-path $2 io.input_dir "$input_run_inv"
toml set --toml-path $2 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $2 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

toml set --toml-path $3 io.input_dir "$input_run_inv"
toml set --toml-path $3 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $3 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

echo $(date -u) "Run inversion stages started" | mail -s "run inv started" beatriz.recinos@ed.ac.uk
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $2 |& tee $path_logs/log_toml_inv_two.txt
OUT=$(tail "$path_logs/log_toml_inv_two.txt")
echo $OUT | mail -s "run inv finish config 2" beatriz.recinos@ed.ac.uk

echo $(date -u) "Run forward stages started" | mail -s "run fwd started" beatriz.recinos@ed.ac.uk
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_forward.py $2 |& tee $path_logs/log_toml_fwd_two.txt
OUT=$(tail "$path_logs/log_toml_fwd_two.txt")
echo $OUT | mail -s "run fwd finish config 2" beatriz.recinos@ed.ac.uk

echo $(date -u) "Run forward stages started" | mail -s "run fwd started" beatriz.recinos@ed.ac.uk
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_forward.py $3 |& tee $path_logs/log_toml_fwd_three.txt
OUT=$(tail "$path_logs/log_toml_fwd_three.txt")
echo $OUT | mail -s "run fwd finish config 3" beatriz.recinos@ed.ac.uk
