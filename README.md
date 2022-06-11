# NEUWON

A toolkit for simulating the brain.

NEUWON is a simulation framework for neuroscience and artificial intelligence
specializing in conductance based models. This software is a modern remake of
the [NEURON](https://www.neuron.yale.edu/neuron/) simulator. It is fast,
accurate, and easy to use.

## Demonstration

![](neuwon/rxd/examples/HH/Staggered_Time_Steps.png)

[TODO: Replace this example with a link to a youtube video showing off a 3D
example. 3D images are much more persuasive than arcane figures. No one is
going to understand what staggered timesteps are, and anyone who does
understand is not going to be impressed by these poor results.]

## Installation and Usage

#### Prerequisites

* [Python 3](https://www.python.org/)
* [An NVIDIA graphics card](https://www.nvidia.com/en-us/geforce/)
* [The CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

#### Installation

```
$ pip install neuwon
```

#### Run the graphical user interface

```
$ python -m neuwon
```

## Methods

### Morphology

NEUWON procedurally generates neurons using the TREES algorithm combined with
the morphological constraints of the ROOTS algorithm.

* One Rule to Grow Them All: A General Theory of Neuronal Branching and Its
Practical Application.  
Cuntz H, Forstner F, Borst A, Hausser M (2010)  
https://doi.org/10.1371/journal.pcbi.1000877

* ROOTS: An Algorithm to Generate Biologically Realistic Cortical Axons and an
Application to Electroceutical Modeling.  
Bingham CS, Mergenthal A, Bouteiller J-MC, Song D, Lazzi G and Berger TW (2020)  
https://doi.org/10.3389/fncom.2020.00013

### NMODL

NMODL is a domain-specific programming language for describing chemical and
protein reactions for neuroscience. NEUWON uses NMODL extensively. NMODL has
been extended many times, and NEUWON further improves the file format.

* [More Information](./NMODL.md)

### Exact Integration

NEUWON uses the exact integration method introduced by (Rotter & Diesmann, 1999)
to simulate diffusion and electric current through passive circuit components.

* Exact digital simulation of time-invariant linear systems with applications to
neuronal modeling.  
Rotter S, Diesmann M (1999)  
https://doi.org/10.1007/s004220050570

### Staggered Integration

Reactions and diffusions interact at staggered time steps, as explained in
chapter 4 of the NEURON book.

* The NEURON Book.  
Carnevale N, & Hines M (2006)  
https://doi.org/10.1017/CBO9780511541612

### Database

NEUWON implements an in-memory database for managing the state of the
simulation. Internally it uses the structure-of-arrays format, and it provides
users with a more familiar object-oriented-programming interface for accessing
the data. The database also provides ancillary features for managing data such
as: error checking, sorting, recording, moving data to/from a graphic card, and
executing python functions on the database using JIT compilation.
