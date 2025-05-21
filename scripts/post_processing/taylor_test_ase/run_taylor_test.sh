#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 -t <toml_file> -i <input_folder> -d <experiment_folder> -r <run_name> -a <alpha_scale> -b <beta_scale>"
    echo ""
    echo "Options:"
    echo "  -t    Path to the TOML configuration file"
    echo "  -i    Path to the input folder"
    echo "  -d    Path to the experiment dir (the one containing output/ and diagnostics/)"
    echo "  -r    Name of the run"
    echo "  -a    scale of alpha perturbation"
    echo "  -b    scale of beta perturbation"
    echo "  -h    Show this help message and exit"
    exit 1
}

# Initialize variables
toml_file=""
input_folder=""
experiment_folder=""
run_name=""
alpha_scale=""
beta_scale=""

# Parse command-line arguments
while getopts "t:i:d:r:a:b:h" opt; do
    case "$opt" in
        t) toml_file="$OPTARG" ;;
        i) input_folder="$OPTARG" ;;
        d) experiment_folder="$OPTARG" ;;
        r) run_name="$OPTARG" ;;
	a) alpha_scale="$OPTARG" ;;
	b) beta_scale="$OPTARG" ;;
        h) show_help ;;
        ?) show_help ;;
    esac
done

# Check if all required arguments are provided
if [[ -z "$toml_file" || -z "$input_folder" || -z "$experiment_folder" || -z "$run_name" || -z "$alpha_scale" || -z "$beta_scale" ]]; then
    echo "Error: Missing required arguments."
    show_help
fi

# Validate file and folders
if [[ ! -f "$toml_file" ]]; then
    echo "Error: TOML file '$toml_file' does not exist."
    exit 1
fi
if [[ ! -d "$input_folder" ]]; then
    echo "Error: Input folder '$input_folder' does not exist."
    exit 1
fi
if [[ ! -d "$experiment_folder" ]]; then
    echo "Error: Experiment folder '$experiment_folder' does not exist."
    exit 1
fi

# Print received inputs
echo "TOML File: $toml_file"
echo "Input Folder: $input_folder"
echo "Experiment Folder: $experiment_folder"
echo "Run Name: $run_name"
echo "Alpha/Beta Scale: $alpha_scale , $beta_scale"

echo "All inputs validated. Proceeding with execution..."


input_dir=$input_folder
output_dir=$experiment_folder/output
diag_dir=$experiment_folder/diagnostics
run_name=$run_name

cp $toml_file ./new_toml.toml 

for N in 3; do

 T=$(bc -l <<< "$N/24")

 toml set --toml-path new_toml.toml io.input_dir "$input_dir"
 toml set --toml-path new_toml.toml io.output_dir "$output_dir"
 toml set --toml-path new_toml.toml io.diagnostics_dir "$diag_dir"

 toml set --toml-path new_toml.toml errorprop.qoi "vaf"
 toml set --toml-path new_toml.toml --to-int errorprop.qoi_vaf_mask_code 2  # PIG
 toml set --toml-path new_toml.toml --to-bool errorprop.qoi_apply_vaf_mask false
 toml set --toml-path new_toml.toml --to-bool errorprop.qoi_vaf_mask_usecode false

 toml set --toml-path new_toml.toml --to-float taylor.scale_alpha $alpha_scale
 toml set --toml-path new_toml.toml --to-float taylor.scale_beta $beta_scale

 toml set --toml-path new_toml.toml --to-float time.run_length $T
 toml set --toml-path new_toml.toml --to-int time.total_steps $N
 toml set --toml-path new_toml.toml --to-int time.num_sens 3

 mpirun -n 24 ./unmute.sh 0 python $FENICS_ICE_BASE_DIR/runs/run_taylor_fwd_inv_noinit.py new_toml.toml  > out${N}_${alpha_scale}_${beta_scale}.txt 2> err${N}_${alpha_scale}_${beta_scale}.txt
 stringbeta=$(grep "beta min_order" out${N}_${alpha_scale}_${beta_scale}.txt);  
 orderbeta=$(echo "$stringbeta" | sed 's/.*min_order: \([0-9.]*\)$/\1/'); 
 stringalpha=$(grep "alpha min_order" out${N}_${alpha_scale}_${beta_scale}.txt);  
 orderalpha=$(echo "$stringalpha" | sed 's/.*min_order: \([0-9.]*\)$/\1/'); 

 echo "Order Alpha = $orderalpha" > res${N}_${alpha_scale}_${beta_scale}.txt
 echo "Order Beta = $orderbeta" >> res${N}_${alpha_scale}_${beta_scale}.txt

done
#done


