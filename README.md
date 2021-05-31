# NEUWON

NEUWON is a simulation framework for neuroscience and artificial intelligence
specializing in conductance based models. This software is a modern remake of
the NEURON simulator. It is accurate, efficient, and easy to use.

## Methods

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

## Usage

#### Prerequisites

* An NVIDIA graphics card
* CUDA
* [Python 3](https://www.python.org/)
* Numba
* CuPY

#### Installation

* `$ pip install neuwon`

#### Documentation

* `$ pydoc neuwon`

#### Model Specification

[todo]

#### Examples

[todo]
