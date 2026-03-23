# Pryonix

Pryonix is A python package for specifying, solving and fitting network based models of prion-like proteinopathies in AD using jax, with built-in functionality for handling connectomes and imaging datasets. 

At the moment, the package focuses on three main pieces of functionality:

- simulating network dynamical models with JAX and Diffrax
- loading and manipulating structural connectomes
- building subject-level and cohort-level PET datasets from ADNI-style tabular data

## Installation

Pryonix currently requires Python 3.11 or greater.

First, download the package: 

```
git clone https://github.com/PavanChaggar/pryonix.git
``` 

Dependencies can be installed using `uv`:

```bash
uv sync
```

or with `pip`:

```bash
pip install .
```

For development and tests:

```bash
uv run pytest
```

## What is currently included


### Models

The current model library includes:

- `NetworkDiffusion`
- `NetworkFKPP`
- `ScaledNetworkFKPP`
- `NetworkATN`
- `ScaledNetworkATN`

These inherit from a common `NetworkModel` base class and are solved through Diffrax.

### Connectomes

The connectome utilities support:
- loading a connectome from a GraphML file
- access to a built-in connectome asset path
- adjacency and Laplacian matrix generation
- filtering weak edges
- slicing to subsets of regions
- reweighting by a custom weight function

### Datasets

The main dataset class currently exposed is `ADNIDataset`, which builds a cohort from a tabular ADNI-style PET dataset. This supports loading SUVR and time values for inputting as initial conditions an ODE model. For example, with the Berekely PET tabular data from ADNI, one can do: 

```

dataset = ADNIDataset.from_dataframe(
    adni_df,
    roi_names=roi_names,
    reference_region="inferiorcerebellum",
)

```
where `adni_df` is the tabular data from ADNI, `roi_names` are the regions of interst for which one wants SUVR data, `reference_region` is the region used to calculate SUVR values. Then one can access the SUVR values and times for a given dataset with: 

```
dataset_suvr = dataset.calc_suvr()
dataset_times = dataset.get_times()
```

## Quick start: Model simulation example

```python
import jax.numpy as jnp
from pryonix.connectomes import Connectome, connectome_path
from pryonix.models import NetworkDiffusion

connectome = Connectome.from_graph_path(connectome_path())
model = NetworkDiffusion(connectome)

n = len(connectome.parc)
y0 = jnp.ones(n)
ts = jnp.linspace(0.0, 10.0, 101)

solution = model.simulate(
    y0,     # initial conditions
    0.1,    # parameters 
    0.0,    # t0
    10.0,   # t final
    ts,     # save at times
)

print(solution.ys.shape)
```

## Current status

Pryonix is currently an early-stage research package.

The dataset functionality is centered on ADNI-style PET tables, and the modelling interface is designed around JAX/Diffrax-based network ODEs. The public API may still evolve as the package grows.