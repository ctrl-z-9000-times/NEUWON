# NEUWON

NEUWON is a simulation framework for neuroscience and artificial intelligence
specializing in conductance based models. This software is a modern remake of
the NEURON simulator. It is fast, accurate, and easy to use.

## Examples

![](neuwon/rxd/examples/HH/Staggered_Time_Steps.png)

## Usage

#### Prerequisites and Installation

* [Python 3](https://www.python.org/)
* [An NVIDIA graphics card](https://www.nvidia.com/en-us/geforce/)
* [The CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* `$ pip install neuwon`
* `$ pydoc neuwon`

#### Model Specification

[todo]

## Simulation Methods

#### Exact Integration

NEUWON uses the exact integration method introduced by (Rotter & Diesmann, 1999)
to simulate diffusion and electric current through passive circuit components.

* Rotter, S., Diesmann, M. Exact digital simulation of time-invariant linear
systems with applications to neuronal modeling. Biol Cybern 81, 381–402 (1999).
https://doi.org/10.1007/s004220050570

#### Staggered Integration

Reactions and diffusions interact at staggered time steps, as explained in
chapter 4 of the NEURON book.

* Carnevale, N., & Hines, M. (2006). The NEURON Book. Cambridge: Cambridge
University Press. https://doi.org/10.1017/CBO9780511541612
