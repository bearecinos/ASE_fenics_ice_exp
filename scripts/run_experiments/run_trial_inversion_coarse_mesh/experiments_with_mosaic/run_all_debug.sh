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
export run_inv_output_dir=$OUTPUT_DIR/06_coarse_mesh_plus_mosaic
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

path_logs=$RUN_CONFIG_DIR/run_trial_inversion_coarse_mesh/experiments_with_mosaic
echo "Logs will be store here:" $path_logs

toml set --toml-path $2 io.input_dir "$input_run_inv"
toml set --toml-path $2 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $2 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

#echo $(date -u) "Run inversion stages started" | mail -s "run inv started" beatriz.recinos@ed.ac.uk
#mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $2 |& tee $path_logs/log_debugg.txt
#OUT=$(tail "$path_logs/log_debugg.txt")
#echo $OUT | mail -s "run inv finish config1" beatriz.recinos@ed.ac.uk

#echo $(date -u) "Run forward stages started" | mail -s "run fwd started" beatriz.recinos@ed.ac.uk
#mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_forward.py $2 |& tee $path_logs/log_tom2_fwd.txt
#OUT=$(tail "$path_logs/log_tom2_fwd.txt")
#echo $OUT | mail -s "run fwd finish config1" beatriz.recinos@ed.ac.uk

echo $(date -u) "Run eigen stages started" | mail -s "run eigendec started" beatriz.recinos@ed.ac.uk
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_eigendec.py $2 |& tee $path_logs/log_tom2_eigen.txt
OUT=$(tail "$path_logs/log_tom2_eigen.txt")
echo $OUT | mail -s "run eigen finish config1" beatriz.recinos@ed.ac.uk

echo $(date -u) "Run error prop stages started" | mail -s "run error prop started" beatriz.recinos@ed.ac.uk
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_errorprop.py $2 |& tee $path_logs/log_tom2_errprop.txt
OUT=$(tail "$path_logs/log_tom2_errprop.txt")
echo $OUT | mail -s "run errp finish config1" beatriz.recinos@ed.ac.uk

echo $(date -u) "Run inv_sigma stages started" | mail -s "run inv sigma started" beatriz.recinos@ed.ac.uk
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_invsigma.py $2 |& tee $path_logs/log_tom2_invsig.txt
OUT=$(tail "$path_logs/log_tom2_invsig.txt")
echo $OUT | mail -s "run inv_sigma finish config1" beatriz.recinos@ed.ac.uk
