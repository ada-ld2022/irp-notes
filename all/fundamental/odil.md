# ODIL

> [!NOTE] 
> Title: ODIL to Solve Forward and Inverse Problems for Partial Differential Equations using Machine Learning Tools
> Author(s): Karnakov et al.

## Summary

- ODIL is a framework that allows the numerical solution of PDEs using machine learning tools.
- Numerical methods are posed as minimisations of discrete residuals solved using gradient based methods.
- The approach is particularly valuable where equations have missing parameters or no data is available to form a well-posed IVP
- The method inherits the accuracy, convergence, and conservation properties of mesh-based discretisations while enabling the usage of modern ML tools such as autodiff.

## Core

### PINN comparison point

PINNs cannot match standard numerical methods for solving classical well-posed problems involving PDEs, but are useful for solving ill-posed and inverse problems. However, the cost of evaluating the solution at one point is proportional to the number of weights, while conventional grid-based methods have a constant cost. In addition, in a PINN, the value of one weight depends on the solution at all points, which does not respect the locality of differential problems. Because PINNs approximate by a non-linear network, the gradient of the residual w.r.t the weights is a dense vector and the Hessian is a dense matrix, where traditional methods allow for sparsity. In short, while PINNs have proven to be a useful tool for approximating the solutions to inverse and non-linear PDE problems, the flexibility, sparsity, accuracy, and convergence properties of traditional PDE constrained optimisation methods leaves much to be desired.

ODIL combines discrete formulations with modern ML tools to extend their scope to ill-posed and inverse problems. As a frameowkr, it follows the discretise-then-differentiate approach that has been formulated as the penalty method (Leeuwen and Herrmann (2015)). This has two key aspects:

1. The discretisation which defines the accuracy, stability, and cnsistency of the method, and
2. The optimisation algorithm to solve the discrete problem

Since the sparsity of the problem is preserved, the optimisation algorithm can use a sparse Hessian and achieve a quadratic convergence rate, which is not possible for gradient-based training of neural networks. This, along with leveraging of modern automatic differentiation tools, makes this method potentially very efficient.

## Methods

In this section, the authors compare ODIL to a finite differncing approach. They look at the wave equation $u_{tt} - u_{xx} = 0$. Instead of solving this equation by time integration from initial conditons, the problem is rewritten as the minimisation of a loss function 

$$
L(u) = 
    \sum_{(i, n)\in \Omega_h} 
    \left(
        \frac { u_{i}^{n+1} - 2 u_{i}^{n} + u^{n-1}_{i} } { \Delta t^2}
        -
        \frac { u_{i+1}^{n} - 2 u_{i}^{n} + u^{n}_{i - 1} } { \Delta x^2}
    \right)^2
    +
    \sum_{(i, n)\in \Gamma_h}
    \left(
        u^n_i - g_i^n
    \right)^2,
$$

Key points:

- $u^n_i$ is a discrete field on a one dimensional Cartesian grid, where $n$ represents the temporal index and $i$ represents the spatial index.
- The second term imposes the boundary conditions $u = g$.
- $\Omega_h$ is the discrete field with a grid spacing of $h$. 
- $\Gamma_h$ is the boundary domain.
- The loss is the sum of the squared discrete PDE residual and the squared boundary residual. 

To apply a gradient-based method such as ADAM or L-BFGS-B, we only need to compute the gradient of the loss, which is also a discrete field. This is because ADAM does not need second order information, and L-BFGS-B approximates the Hessian implictly.

To use Newton's method, however, recall we need the Hessian $H$ (or an approximation in Gauss-Newton methods) to compute updates as

$$u_{k+1} = u_{k} = -H^{-1}\nabla L.$$

Computing and inverting the full Hessian is expensive, but the deeper problem is that for a general non-linear loss, the Hessian is not guaranteed to be PD, which means the Newton step might not even be in a descent direction.

To address this, we can use a sum of squares approach. When the loss has the form $L(u) = \|F(u)\|^2_2 + \|G(u)\|^2_2$, if you linearise the operators around the current solution $u_s$,

$$F[u] \approx F[u_s] + J_F(u - u_s),$$

where $J_F$ is the Jacobian of $F$, then the Hessian becomes

$$H \approx J^T_F J_F + J^T_G J_G.$$

This is the Gauss-Newton approximation, which has 3 critical properties:

1. Guaranteed PSD, so the step is always in a valid direction
2. $J$ is typically sparse, and $H$ inherits that sparsity and can be inverted directly
3. Cheap to form: you only need the Jacobian

These points make Newton's method feasible, but not neccesarily superior. It offers faster local convergence, exploits sparsity directly, and has deterministic and interpretable steps, but if the PDE is highly non-linear and the residuals are large, the approximation degrades. ADAM and L-BFGS-B are more robust across a wider range of conditions. Papers typically use one of these for a warmup and then switch to Newton's for faster local convergence.

Continuing on, the loss becomes explicitly quadratic in $u$:

$$
L^s(u^(s+1)) = 
    \|\mathbf{F}^s + \left( \frac {\delta \mathbf{F}^s} {\partial u} \right)(u^{s+1} - u^s)\|^2_2
    +
    \|\mathbf{G}^s + \left( \frac {\delta \mathbf{G}^s} {\partial u} \right)(u^{s+1} - u^s)\|^2_2,
$$

where $\mathbf{F}^s$ denotes $\mathbf{F}[u^s]$ and $\mathbf{G}^s$ denotes $\mathbf{G}[u^s]$.

## Practical takeaways

ODIL's key insight is to treat the PDE solution as the direct optimisation variable. We are not learning weights, we are finding the field which minises a discrete loss that encodes PDE residuals.

A summary of practical steps are as follows:

1. Discretise on a grid.
2. Define the discrete operators $F$ (the PDE residual) and $G$ (boundary/ initial conditions). They must be differentiable w.r.t. $u$.
3. Define the loss.
4. Choose and apply an optimiser
    a. for ADAM or L-BFGS-B, you need $\nabla L$ w.r.t $u$ at each step. Since $F$ and $G$ are discrete array operations, we can get this via automatic differentiation by manually deriving finite difference stencil gradients. We hand the loss and gradient to the optimiser and iterate until convergence
    b. For Gauss-Newton, compute the Jacobians $J_F$ and $J_G$. These are sparse matrices which describe how much each residual element depends on each grid point. Form the approximate Hessian and the gradient, solve the sparse linear system $H \delta u = -g$ for the next update step, and apply.
5. Handle the inverse problem. ODIL handles inverse problems where some parameters $\theta$ of the PDE are unknown. The optimisation variable becomes $(u, \theta)$ jointly. $F[u]$ becomes $F[u, \theta]$, and the loss and gradient/Jacobian computations extend naturally to include $\theta$. The same optimisers now apply.
6. Monitor the loss and individual residuals $F$ and $G$ separately to see whether we are failing on the PDE interior or BCs. Check $u$ is physically plausible and, for Gauss-Newton, check that $H$ has a good conditioning number, else we might need to apply regularisation or a better grid.

In psuedocode of sorts:

```
Initialize u (e.g. zeros, random, or a coarse guess)
    ↓
Evaluate F[u], G[u]  →  compute L(u)
    ↓
Compute gradient (Adam/L-BFGS-B) or Jacobian+linear solve (Gauss-Newton)
    ↓
Update u
    ↓
Repeat until ‖F[u]‖ and ‖G[u]‖ are sufficiently small
```

### FWI integration

FWI seeks to recover material properties by minimising the misfit between observed and synthetic residuals generated by solving the wave equation. The full workflow is:

1. Guess an initial model
2. Solve the forward wave equation to get synthetics
3. Compute the misfit
4. Solve the adjoint for the gradient
5. Update the model with an optimiser

The key difficulty is that the forward solve and inversion are decoupled: one must fully solve the wave equation at each iteration to evaluate the misfit and its gradient.

ODIL collapses this into a single joint optimisation over both the wavefield and the velocity model simultaneously. We never explicitly solve the forward problem, instead the functional enforces it. The optimisation variables are now $(u, c)$ and the loss looks something like

$$
L(u, c) = 
    \underbrace{\|F[u, c]\|^2_2}_\text{wave eq. residual}
    +
    \underbrace{\|G[u, c]\|^2_2}_\text{bdry/ src conditions}
    +
    \underbrace{\|Ru - d_\text{obs}\|^2_2}_\text{data residual}.
$$

Here, $F$ is the discretised wave equation residual, $G$ encodes the initial conditions, and $Ru$ is a restriction operator applied to the wavefield.

The key steps then become something like:

1. Discretise the joint space
2. Build the discrete operators $F$ as a finite difference stencil for thre wave equation and $G$ for the ABCs and source injection, and then the Jacobian of the data term is just $R$.
3. Form the Jacobians $\delta F/ \delta u,\ \delta G/ \delta c,\ \delta Ru/ \delta u$.These are nice and sparse.
4. Cycle or joint update: either fix $c$ and update $u$ then fix $u$ and update $c$ or optimise over $u, c$ simultaneously at each step.

The core strengths of this approach are:

- FWI gets trapped in local minima due to a bad initial model (cycle skipping). ODIL optimises jointly, so the wavefield is free to adapt both the data and PDE simultaneously, which can allow for more robust loss landscape navigation.
- ODIl does not require an adjoint solve, so this does not need to be implemented at all, saving time and computational work.
- Since $F$ is being discretised directly, there is no hiddne inconsistency between forward solver and gradient computation.

The key challenges are memory consumption, source encoding, and joint initialisation.
