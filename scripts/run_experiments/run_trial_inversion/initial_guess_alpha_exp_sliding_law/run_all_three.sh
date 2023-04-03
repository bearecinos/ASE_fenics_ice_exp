#!/bin/bash

# Create and define your run_input directory for the workflow run
input_run_inv=$INPUT_DIR/input_run_workflow

# Create output directory for inversion output
export run_inv_output_dir=$OUTPUT_DIR/04_sliding_exp
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

path_logs=$RUN_CONFIG_DIR/run_trial_inversion/initial_guess_alpha_exp_sliding_law
echo "Logs will be store here:" $path_logs

toml set --toml-path $2 io.input_dir "$input_run_inv"
toml set --toml-path $2 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $2 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

toml set --toml-path $3 io.input_dir "$input_run_inv"
toml set --toml-path $3 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $3 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

toml set --toml-path $4 io.input_dir "$input_run_inv"
toml set --toml-path $4 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $4 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

echo $(date -u) "Run inversion stages started"

mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $2 |& tee $path_logs/log_tom2_inv.txt
OUT=$(tail "$path_logs/log_tom2_inv.txt")
echo $OUT | mail -s "run inv finish config1" beatriz.recinos@ed.ac.uk

mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $3 |& tee $path_logs/log_tom3_inv.txt
OUT=$(tail "$path_logs/log_tom3_inv.txt")
echo $OUT | mail -s "run inv finish config2" beatriz.recinos@ed.ac.uk

mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $4 |& tee $path_logs/log_tom4_inv.txt
OUT=$(tail "$path_logs/log_tom4_inv.txt")
echo $OUT | mail -s "run inv finish config3" beatriz.recinos@ed.ac.uk

