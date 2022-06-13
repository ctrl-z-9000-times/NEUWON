
# NMODL

NMODL is a domain-specific programming language for describing chemical and
protein reactions for neuroscience. NEUWON uses NMODL extensively. NMODL has
been extended many times, and NEUWON further improves the file format.

### Changes to the specification

* "USEION" statements now allows access all chemical species regardless of
  their electric charge, and their intracellular and extracellular
  concentrations can be read and written via the standardly named variables.
  Previously, accessing this data required a "POINTER" statement.

* "USEION" statements now provide WRITE access to conductance data.  
  The syntax is "`USEION x WRITE gx`" and the units of `gx` are ???.  
  Multiple mechanisms can write to the same species conductance without issue.

* "USEION" statements now provide a way to create nonspecific conductances, by
   assigning to the ion's equilibrium potential.
   Use the syntax "`USEION x WRITE ex`" and specify `ex` in your "PARAMETERS"
   block in units of mV.

* All mechanisms now have a magnitude and standard way of accessing it.
    * POINT_PROCESS ...
    * SUFFIX ...
    * Access other mechanisms magnitudes ...

* NMODL now exposes less information about the model's internal state to the
  simulator. These changes enabled significant performance optimizations.

    * "FUNCTION" and "PROCEDURE" blocks defined in NMODL can now only be called
       from within NMODL files. They are not accessible in NEUWON. The only
       methods accessible from NEUWON are to create, destroy, and advance the
       state of mechanisms.

    * "ASSIGNED" variables are now freed from memory at the end of the
      BREAKPOINT block, for improved memory consumption.  

### History of NMODL

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

### Documentation

* [NEURON's documentation on NMODL](https://nrn.readthedocs.io/en/latest/python/modelspec/programmatic/mechanisms/nmodl.html)

 * How to expand NEURON’s library of mechanisms  
Chapter 9 of The NEURON book.  
Carnevale, N. T., & Hines, M. L. (2006).

### Where to find NMODL files?

* NEUWON  
[I intend to keep a fairly complete set of models in the project. I should state
that my goal is to make a good toolkit, and the specific models are subject to
change.]

* [ModelDB](https://senselab.med.yale.edu/ModelDB/)

* [ChannelPedia](https://channelpedia.epfl.ch/)

* Automatic conversion from NeuroML
