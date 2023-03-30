n_layers = 1
mask_angles = 0.0:0.01:0.25

for angle in mask_angles
    cmd = `srun --gres=gpu:1 -t 00:20:00 python test_script.py --n_layers $n_layers --mask_angle $angle`
    run(cmd)
    sleep(0.1)