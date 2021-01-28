# NEUWON

NEUWON is a simulation framework for neuroscience and artificial intelligence
specializing in conductance based models. This software is a modern remake of
the NEURON simulator. It is accurate, efficient, and easy to use.

## Model Architecture

#### Physical Structures

This section describes how NEUWON represents the physical shape of a neuron.

Neurons are composed of many cylindrical segments which are connected into a
tree structure. The tips of the cylinders serve as tracking points for all
cellular processes: intracellular, extracellular, and membrane-related. The
extracellular space is partitioned into Voronoi cells which are centered on the
tracking points. NEUWON provides the following geometric information about each
tracking point:

* Tree Structure
    + Parent Segment, unless segment is root of tree
    + Child Segments, list
    + Distance between segment and parent segment
* Segment Properties
    + Diameter
    + Cross-sectional Area
    + Membrane Surface Area
    + Intracellular Volume
* Extracellular Space Properties
    + Volume
    + Adjacent Tracking Points
        - Distance between points
        - Surface Area of border

#### Reactions & Diffusions

The brain uses chemical reactions to implement arbitrary logic. Large proteins
can maintain a persistent state, and when embedded in the cell membrane can
sense and control the electrical potential across the membrane. Chemicals can
diffuse through the interior of a neuron, through the extracellular space
between cells, and across cell membranes. Electrically charged ions flow
throughout neurons; a phenomenon which is modeled with an equivalent electrical
circuit.

[todo state which things NEUWON covers and which things the user needs to
specify and integrate over time and w/ neuwon]

## Integration Methods

#### Exact Integration

NEUWON uses the exact integration method introduced by (Rotter & Diesmann, 1999).
[todo state which components use this method]

* Rotter, S., Diesmann, M. Exact digital simulation of time-invariant linear
systems with applications to neuronal modeling. Biol Cybern 81, 381–402 (1999).
https://doi.org/10.1007/s004220050570

#### Staggered Integration

Reactions and diffusions interact at staggered time steps, as explained in
chapter 4 of the NEURON book.

[todo citation]

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

[todo: discuss prefix-less unit system]

#### Examples

[todo]
