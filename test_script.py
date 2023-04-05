import haiku as hk
import jax.random as jrnd
import jax.numpy as jnp
import numpy as np
import optax
import pickle as p
import argparse
import pandas as pd

from phasor_jax.modules import *
from phasor_jax.utils import *
from phasor_jax.training import *

"""
Get command line arguments.
"""
parser = argparse.ArgumentParser(description="Test spiking execution of phasor network.")
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--dataset", type=str, default="fashion_mnist")
parser.add_argument("--n_batch", type=int, default=128)
parser.add_argument("--prng_seed", type=int, default=42)
parser.add_argument("--n_batches", type=int, default=1000)
parser.add_argument("--params_file", type=str, default=None)
parser.add_argument("--mask_angle", type=float, default=0.0)
parser.add_argument("--cross_inhibit", type=float, default=0.0)
parser.add_argument("--random_removal", type=float, default=0.0)

args = parser.parse_args()
n_layers = args.n_layers
dataset = args.dataset
n_batch = args.n_batch
prng_seed = args.prng_seed
n_batches = args.n_batches
params_file = args.params_file
mask_angle = args.mask_angle
cross_inhibit = args.cross_inhibit
random_removal = args.random_removal
#add more time for deeper layers to propagate
t_exec = 9.75 + 0.25 * n_layers

"""
Load the dataset.
"""
train_ds, x_train, y_train = load_dataset(dataset, split="train", is_training=True, batch_size=n_batch)
train = iter(train_ds)

#Load the testing dataset
test_ds, x_test, y_test = load_dataset(dataset, split="test", is_training=False, repeat = False, batch_size=n_batch)
test = iter(test_ds)

"""
Define the model.
"""

def mlp(x, 
           n_layers: int = 1,
           n_hidden: int = 128,
           vsa_dimension: int = 1024,
           spiking: bool = False,
           repeats: int = 3,
           is_training: bool = False,
           **kwargs):
    """
    Simple MLP model with scalable number of hidden layers
    """
    
    x = scale_mnist(x)
    #project into VSA
    x = ProjectAll(vsa_dimension)(x)
    x = layer_norm(x)

    if spiking:
        x = phase_to_train(x, repeats=repeats)
    
    outputs = []
    for i in range(n_layers):
        x = conv_1d(n_hidden)(x, spiking=spiking, **kwargs)
        outputs.append(x)

    x = conv_1d(10)(x,  spiking=spiking, **kwargs)
    outputs.append(x)

    #only return the single output if training
    if is_training:
        return x

    if spiking: 
        p = train_to_phase(x)

        return p, outputs
    else:

        return x, outputs

"""
Initialize the model
"""
#declare the model as a transformation
model = hk.transform(mlp)

#split the key and use it to create the model's initial parameters
key = jrnd.PRNGKey(prng_seed)
key, subkey = jrnd.split(key)
params = model.init(subkey, x_train[0:10,...], n_layers = n_layers)

"""
Train the model if necessary
"""
if params_file is None:
    #create an instance of the RMSprop optimizer
    opt = optax.rmsprop(0.001)
    loss_fn = lambda yh, y: quadrature_loss(yh, y, num_classes=10)

    params_t, losses = train_model(model, 
                                key, 
                                params = params, 
                                dataset = train, 
                                optimizer = opt, 
                                loss_fn = loss_fn, 
                                batches = n_batches,
                                n_layers = n_layers)
    
else:
    params_t = p.load(open(params_file, "rb"))

"""
Test performance
"""

all_results = {}
all_results["mask angle"] = mask_angle
all_results["cross inhibit"] = cross_inhibit
all_results["random_removal"] = random_removal

#define a lambda to compute accuracy we can dispatch over batches
eval_fn = lambda x: model.apply(params_t, key, x, n_layers = n_layers, mask_angle = mask_angle)

#define a lambda to compute the spiking equivalent
if mask_angle > 0.0:
    spike_filter = lambda x, shp: inhibit_midpoint(x, mask_angle=mask_angle)
    filename = "phasor_" + str(n_layers) + "_layers_angle_" + str(mask_angle) + ".p"
elif cross_inhibit > 0.0:
    spike_filter = lambda x, shp: inhibit_field(x, cross_inhibit, shp)
    filename = "phasor_" + str(n_layers) + "_layers_inhibit_" + str(cross_inhibit) + ".p"
elif random_removal > 0.0:
    spike_filter = lambda x, shp: inhibit_random(x, random_removal)
    filename = "phasor_" + str(n_layers) + "_layers_random_" + str(random_removal) + ".p"
else:
    spike_filter = None
    filename = "phasor_" + str(n_layers) + "_layers.p"

eval_fn_spk = lambda x: model.apply(params_t, key, x, n_layers = n_layers, spike_filter = spike_filter, spiking = True)


def test_normal():
    #compute the test set accuracy
    result = [eval_fn(b['image']) for b in iter(test_ds)]
    predictions = jnp.concatenate([r[0] for r in result])
    #get the overall accuracy
    accuracy = accuracy_quadrature(predictions, y_test)
    acc = np.mean(accuracy)
    #get the sparsity at each layer & save
    batch_usage = np.stack([np.array(list(map(matrix_usage, r[1]))) for r in result])
    avg_usage = np.mean(batch_usage, axis=0)
    return acc, avg_usage

acc, avg_usage = test_normal()

print("Test accuracy: ", acc)
all_results["accuracy"] = acc
all_results["matrix usage"] = avg_usage


def pad_outputs(phases):
    """
    Pad spiking outputs to a consistent shape for concatenation.
    """
    shapes = np.array([p.shape[2] for p in phases])
    max = np.max(shapes)
    padding = max -  shapes
    
    #pad out the cycles since some spiking evaluations may have fewer
    for i in range(len(phases)):
        if padding[i] > 0:
            pad_fn = lambda x: np.pad(x, ((0, 0), (0, 0), (0, padding[i])))
            phases[i] = pad_fn(phases[i])

    return np.concatenate(phases, axis=0)


def test_spiking():
    #repeat the process with spiking output
    result_spk = [eval_fn_spk(b['image']) for b in tqdm(iter(test_ds))]
    predictions_spk = pad_outputs([r[0] for r in result_spk])
    #get the overall accuracy
    accuracy_spk = accuracy_quadrature(predictions_spk, y_test)
    acc_spk = np.mean(accuracy_spk, axis=0)
    #get the sparsity at each layer & save
    batch_usage_spk = np.stack([np.array(list(map(spiking_rate, r[1]))) for r in result_spk])
    avg_usage_spk = np.mean(batch_usage_spk, axis=0)
    return acc_spk, avg_usage_spk

acc_spk, avg_usage_spk = test_spiking()

print("Spiking accuracy: ", acc_spk)
all_results["accuracy_spiking"] = acc_spk
all_results["firing rates"] = avg_usage_spk


with open(filename, 'wb') as file:
    p.dump(all_results, file)