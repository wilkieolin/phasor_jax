n_layers = 2
mask_angles = 0.022:0.01:0.23

#train the model
n_batches = 1000
cmd = `srun --gres=gpu:1 -t 00:20:00 python train_script.py --n_batches $n_batches --n_layers $n_layers`
run(cmd)

params_file = "params_" * string(n_layers) * "_layers.p"
#test it over the range of mask angles
for angle in mask_angles
    cmd = `sbatch --gres=gpu:1 -t 00:20:00 python test_script.py --n_layers $n_layers --mask_angle $angle --params_file $params_file`
    run(cmd)
    sleep(0.1)
end
