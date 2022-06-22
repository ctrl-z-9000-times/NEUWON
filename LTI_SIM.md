# LTI_SIM

Simulator for Linear Time-Invariant Kinetic Models using the NMODL file format.

For more information see:
 *  Exact digital simulation of time-invariant linear systems with applications
    to neuronal modeling. Rotter S, Diesmann M (1999). 
    https://doi.org/10.1007/s004220050570

### Usage

```
$ python ./lti_sim/ --help
```

Chemical concentration inputs should be preprocessed using the logarithmic option.

### Methods

Linear Time-Invariant systems have exact solutions, using the matrix-exponential
function. The result is a "propagator" matrix which advances the state of the
system by a fixed time step. To advance the state simply multiply the
propagator matrix and the current state vector.

However computing the matrix exponential is time-consuming and it is actually a
function of the inputs to the system, so using this method naively at run time
is prohibitively slow.

---

This program computes the propagator matrix for every possible combination of
inputs. It then reduces all of those exact solutions into an approximation
which is optimized for both speed and accuracy.

The approximation is structured as follows:
1. The input space is divided into evenly spaced bins.
2. Each bin contains a polynomial approximation of the function.  
   All polynomial have the same degree, just with different coefficients.

The optimization proceeds as follows:

1. Start with an initial configuration, which consists of a polynomial degree and a
number of input bins.  
For example models with one input start with a cubic polynomial and 10 input bins.

2. Determine the number of input bins which yields the target accuracy.  
   The accuracy is directly proportional to the number of input bins.

3. Measure the speed performance of the configuration by running a small but
realistic benchmark.

4. Experiment with different polynomials to find the fastest configuration which
meets the target accuracy.  
Use a simple hill-climbing procedure to find the first local maxima of
performance.
