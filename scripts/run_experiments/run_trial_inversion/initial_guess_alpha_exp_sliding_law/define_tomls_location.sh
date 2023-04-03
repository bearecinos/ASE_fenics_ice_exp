#!/bin/bash

tomls_main_dir=$RUN_CONFIG_DIR/run_trial_inversion/initial_guess_alpha_exp_sliding_law

echo $tomls_main_dir

toml1=$tomls_main_dir/ase_alpha_init_2_budd.toml
toml2=$tomls_main_dir/ase_alpha_init_5_budd.toml
toml3=$tomls_main_dir/ase_alpha_init_10_budd.toml

#nohup bash $RUN_CONFIG_DIR/run_trial_inversion/initial_guess_alpha_exp_sliding_law/run_all_three.sh 24 toml1 toml2 toml3 >/dev/null 2>&1