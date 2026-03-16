(reference:overview)=
# Overview

Ptychographic reconstruciton is often solved as an optimization problem using gradient descent methods. Typical ptychographic reconstruction packages (e.g. *PtychoShelves*, *PtyPy*, *py4DSTEM*) use an analytically derived or approximated gradients and apply them for the updates, while *PtyRAD* utilizes automatic differention (AD) to automatically calculate the needed gradients.

The main difference is that automatic differentiation allows simpler implementation for adding and modifying new optimizable variables. For typcial packages utilize analytical gradients, adding a new optimizable variable (like adding probe position correction or adaptive beam tilt) requires deriving the corresponding gradient with respect to the objective (loss) funciton. Manually deriving the gradients for new variables can be a tedious and daunting task. On the other hand, automatic differentiation provides instant gradients as long as a differentialble forward model is provided, and flexible control over the optimizable variables including optimizing the amplitude and phase of the object individually.

## Why PtyRAD

### High performance

### Flexible
