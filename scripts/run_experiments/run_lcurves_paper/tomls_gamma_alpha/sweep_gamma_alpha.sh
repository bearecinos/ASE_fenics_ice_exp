path_logs=$RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha
echo "Logs will be store here:" $path_logs

mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/ase_template_ga_1e-04.toml |& tee log__ga_1e-04.txt
OUT=$(tail "$path_logs/log__ga_1e-04.txt")
echo $OUT | mail -s "run inv finish config1e-04" beatriz.recinos@ed.ac.uk

mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/ase_template_ga_1e-03.toml |& tee log__ga_1e-03.txt
OUT=$(tail "$path_logs/log__ga_1e-03.txt")
echo $OUT | mail -s "run inv finish config1e-03" beatriz.recinos@ed.ac.uk

mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/ase_template_ga_1e-02.toml |& tee log__ga_1e-02.txt
OUT=$(tail "$path_logs/log__ga_1e-02.txt")
echo $OUT | mail -s "run inv finish config1e-02" beatriz.recinos@ed.ac.uk

mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/ase_template_ga_1e-01.toml |& tee log__ga_1e-01.txt
OUT=$(tail "$path_logs/log__ga_1e-01.txt")
echo $OUT | mail -s "run inv finish config1e-01" beatriz.recinos@ed.ac.uk

mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/ase_template_ga_1e+00.toml |& tee log__ga_1e+00.txt
OUT=$(tail "$path_logs/log__ga_1e+00.txt")
echo $OUT | mail -s "run inv finish config1e+00" beatriz.recinos@ed.ac.uk

mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/ase_template_ga_1e+01.toml |& tee log__ga_1e+01.txt
OUT=$(tail "$path_logs/log__ga_1e+01.txt")
echo $OUT | mail -s "run inv finish config1e+01" beatriz.recinos@ed.ac.uk

mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/ase_template_ga_1e+02.toml |& tee log__ga_1e+02.txt
OUT=$(tail "$path_logs/log__ga_1e+02.txt")
echo $OUT | mail -s "run inv finish config1e+02" beatriz.recinos@ed.ac.uk

mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/ase_template_ga_1e+03.toml |& tee log__ga_1e+03.txt
OUT=$(tail "$path_logs/log__ga_1e+03.txt")
echo $OUT | mail -s "run inv finish config1e+03" beatriz.recinos@ed.ac.uk

mpirun -n 24 python $RUN_CONFIG_DIR/run_lcurves/run_lcurves_inv.py $RUN_CONFIG_DIR/run_lcurves_paper/tomls_gamma_alpha/ase_template_ga_1e+04.toml |& tee log__ga_1e+04.txt
OUT=$(tail "$path_logs/log__ga_1e+04.txt")
echo $OUT | mail -s "run inv finish config1e+04" beatriz.recinos@ed.ac.uk
