n_layers = 3
mask_angles = 0.0:0.01:0.25

#train the model
n_batches = 1000
params_file = "params_" * string(n_layers) * "_layers.p"

if !isfile(params_file)
    cmd = `python train_script.py --n_batches $n_batches --n_layers $n_layers`
    run(cmd)
end

#test it over the range of mask angles
for angle in mask_angles
    cmd = `python test_script.py --n_layers $n_layers --mask_angle $angle --params_file $params_file`
    run(cmd)
end