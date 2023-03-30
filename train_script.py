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

args = parser.parse_args()
n_layers = args.n_layers
dataset = args.dataset
n_batch = args.n_batch
prng_seed = args.prng_seed
n_batches = args.n_batches
params_file = args.params_file

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
        best_cycle = dphase_postmax(p)
        p = p[:,:,best_cycle]

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
Train the model
"""
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



filename = "params_" + str(n_layers) + "_layers"
with open(filename, 'wb') as file:
    p.dump(file, params_t)