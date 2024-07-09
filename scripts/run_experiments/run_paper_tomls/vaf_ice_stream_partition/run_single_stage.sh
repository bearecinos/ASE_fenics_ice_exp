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
export run_inv_output_dir=$OUTPUT_DIR/06_vaf_partition
if [ ! -d $run_inv_output_dir ]
then
  echo "Creating run directory $run_inv_output_dir"
  mkdir $run_inv_output_dir
else
  echo "Directory is $run_inv_output_dir already exist"
fi

toml_1=$RUN_CONFIG_DIR/run_paper_tomls/vaf_ice_stream_partition/ase_measures-vel-mosaic_C0a2-2_L0a-7100_C0b2-17_L0b-1000_THW.toml
toml_2=$RUN_CONFIG_DIR/run_paper_tomls/vaf_ice_stream_partition/ase_measures-vel-mosaic_C0a2-2_L0a-7100_C0b2-17_L0b-1000_SPK.toml
toml_3=$RUN_CONFIG_DIR/run_paper_tomls/vaf_ice_stream_partition/ase_measures-vel-mosaic_C0a2-2_L0a-7100_C0b2-17_L0b-1000_PIG.toml

path_logs=$RUN_CONFIG_DIR/run_paper_tomls/vaf_ice_stream_partition
echo "Logs will be store here:" $path_logs

toml set --toml-path $toml_1 io.input_dir "$input_run_inv"
toml set --toml-path $toml_1 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $toml_1 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

toml set --toml-path $toml_2 io.input_dir "$input_run_inv"
toml set --toml-path $toml_2 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $toml_2 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

toml set --toml-path $toml_3 io.input_dir "$input_run_inv"
toml set --toml-path $toml_3 io.output_dir "$run_inv_output_dir/output"
toml set --toml-path $toml_3 io.diagnostics_dir "$run_inv_output_dir/diagnostics"

echo $(date -u) "Run inversion stages started" | mail -s "run inv started" beatriz.recinos@ed.ac.uk
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $toml_1 |& tee $path_logs/log_toml_inv_1.txt
OUT=$(tail "$path_logs/log_toml_inv_1.txt")
echo $OUT | mail -s "run inv finish config 1" beatriz.recinos@ed.ac.uk

echo $(date -u) "Run inversion stages started" | mail -s "run inv started" beatriz.recinos@ed.ac.uk
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $toml_2 |& tee $path_logs/log_toml_inv_2.txt
OUT=$(tail "$path_logs/log_toml_inv_2.txt")
echo $OUT | mail -s "run inv finish config 2" beatriz.recinos@ed.ac.uk

echo $(date -u) "Run inversion stages started" | mail -s "run inv started" beatriz.recinos@ed.ac.uk
mpirun -n $1 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $toml_3 |& tee $path_logs/log_toml_inv_3.txt
OUT=$(tail "$path_logs/log_toml_inv_3.txt")
echo $OUT | mail -s "run inv finish config 3" beatriz.recinos@ed.ac.uk


