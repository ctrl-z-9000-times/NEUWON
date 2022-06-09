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

* The "USEION" statement now allows access all chemical species regardless of
  their electric charge, and their intracellular and extracellular
  concentrations can be read and written via the standardly named variables.
  Previously, accessing this data required a "POINTER" statement.

* "USEION" statements now provide WRITE access to conductance data.  
  The syntax is "`USEION x WRITE gx`" and the units of `gx` are ???.  
  Multiple mechanisms can write to the same species conductance without issue.

* All mechanisms now have a magnitude and standard way of accessing it
  using "POINTER" statements. Mechanisms can access the magnitudes of other
  mechanisms too.

* NMODL now exposes less information about the model's internal state to the
  simulator. These changes enabled significant performance optimizations.

    * "FUNCTION" and "PROCEDURE" blocks defined in NMODL can now only be called
       from within NMODL files. They are not accessible in NEUWON. The only
       methods accessible from NEUWON are to create, destroy, and advance the
       state of mechanisms.

    * "ASSIGNED" variables are now freed from memory at the end of the
      BREAKPOINT block, for improved memory consumption.  

* "ASSIGNED", "RANGE", and "LOCAL" statements no longer mean anything and are
   not necessary.

    * "RANGE" variables [what did they do again?]

    * Local variable are automatically created when a new variable name is
      assigned to, like it is in python.

#### History of NMODL

NMODL is an extension of the model description language developed for SCoP;
published in:

* A block organized model builder.  
Kohn, M. C., Hines, M. L., Kootsey, J. M., & Feezor, M. D. (1994).  
https://doi.org/10.1016/0895-7177(94)90190-2

The NEURON program extended the SCoP language with many neuroscience specific
features to become NMODL.

* Expanding NEURON's repertoire of mechanisms with NMODL.  
Hines, M. L., & Carnevale, N. T. (2000).  
https://doi.org/10.1162/089976600300015475

#### Documentation

* [NEURON's documentation on NMODL](https://nrn.readthedocs.io/en/latest/python/modelspec/programmatic/mechanisms/nmodl.html)

 * How to expand NEURON’s library of mechanisms  
Chapter 9 of The NEURON book.  
Carnevale, N. T., & Hines, M. L. (2006).

#### Where to find NMODL files?

* NEUWON  
[I intend to keep a fairly complete set of models in the project. I should state
that my goal is to make a good toolkit, and the specific models are subject to
change.]

* [ModelDB](https://senselab.med.yale.edu/ModelDB/)

* [ChannelPedia](https://channelpedia.epfl.ch/)

* Automatic conversion from NeuroML

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
