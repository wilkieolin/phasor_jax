n_layers::Int = 1
mask_angles = 0.525:0.025:0.95
cross_inhibits = 0.15:0.025:0.60
random_removals = 0.05:0.05:0.95
use_slurm::Bool = false

cmds = []

#train the model
n_batches = 1000
params_file = "params_" * string(n_layers) * "_layers.p"

if use_slurm
    prefix = ["srun",
             "--gres=gpu:1",
             "-t 00:30:00",
             "python"]
else
    prefix = ["python"]
end

if !isfile(params_file)
    train = ["train_script.py", "--n_batches $n_batches", "--n_layers $n_layers"]
    run(Cmd([prefix..., train...]))
end

#test it over the range of mask angles
# for angle in mask_angles
#     mask_test = ["test_script.py", "--n_layers", string(n_layers), 
#                                 "--mask_angle", string(angle), 
#                                 "--params_file", params_file]
#     run(Cmd([prefix..., mask_test...]))
# end

#test it over the range of inhibition times
for cross_inhibit in cross_inhibits
    inhibit_test = ["test_script.py",
                    "--n_layers", string(n_layers),
                    "--cross_inhibit", string(cross_inhibit),
                    "--params_file", params_file]
    run(Cmd([prefix..., inhibit_test...]))
end

# #test it over the range of inhibition times
# for random_removal in random_removals
#     random_test = ["test_script.py",
#                     "--n_layers", string(n_layers),
#                     "--random_removal", string(random_removal),
#                     "--params_file", params_file]
#     run(Cmd([prefix..., random_test...]))
# end