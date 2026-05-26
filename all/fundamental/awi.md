# AWI

> [!NOTE]
> Title: Adaptive Waveform Inversion
>
> Author(s): Guasch and Warner, 2016
## Intro

- FWI produces a spurious model if the starting model is too far removed from the true model, demonstrating cycle skipping
- AWI appears to be immune to this
- AWI applies least squares convolutional filters to transform the predicted data into the observed data. The problem is formulated such that the subsufrace model forces these Wiener fiters towards zero lag delta functions.
- A Wiener filter is an optimal linear filter which minimises the mean square error between an estimated signal and a desired signal in the presence of noise. A zero-lag delta function is a init spike centered at t = 0.
- 'Forcing a Wiener filter towards a zero lag delta function' means that the desired output is a perfectly sharp spike. We want zero lag so no time delay is introduced, and a delta like shape so all energy is concentrated at one sample.
- AWI has a similar cost and covergence behaviour to FWI.


- Conventional FWI minimises the sum of squares of the sample-by-sample differences between predicted and observed data, but can be configured such that it minises the zero lag cross correlation between them.
- FWI can avoid cycle skipping by 
    - beggining the inversion at low frequencies
    - starting from an accurate model
    - limiting the inversion to shallow depths and short travel times
    - rigorous data quality control

## AWI

AWI seeks to find a suite of data-adaptive matching filters that transform one data set into the other. Since the filters depend on the predicted and observed data, they also depend on the true and current model. If the true and current model are the same, the filters must degenerate to identity filters which do not alter the input data.

A matching filter $f$ is one that is designed to transform one signal into another:

$$d_\text{pred}f \ast \approx d_\text{obs}. $$

We are asking "what must I convolve my predicted data with to make it look like the observed data?" This is basically then a Wiener filter problem in which we seek to find $f$ such that we minimise the least squares difference between $d_\text{pred} \ast f$ and $d_\text{obs}$. The 'forcing towrds a zero-lag delta' is the converge criterion: we compare the computed matching filter to an ideal delta.

This is the basis of AWI: *the waveform inversion is setup to minimise the misfit between the filters required to match the two data sets and the ideal identity filter.*

Some core ideas which will come up:

- AWI avoids cycle skipping using matching filters
- AWI reformulates the underlying problem so predicted data need not obey a physical wave equation
- The Wiener filter coefficients add an additional dimenion to the subsurface model, and by treating the entire wavefield as part of the unknown model space in conjunction, AWI massively expands the dimensionaity of the unknown model, which appears to be what helps with cycle skipping

## Formulation

### Objective function

#### FWI

FWI seeks to minimise

$$f = \frac 1 2 \|\mathbf{p - d}\|^2,$$

i.e., half the sum of the squares of the difference between the predicted and observed data.

#### Filter matching

AWI defines a convolutional matching filter $\bf w$ which, when convolved with the predicted data trace $\bf p$, propvides the best least-squares match to the observed data trace $\bf d$. The coefficients which define the filter are those that minimise the function $g$ given by

$$g = \frac 1 2 \| \mathbf{Pw - d} \|^2,$$

where $\bf P$ is a Toeplitz matrix (constant diagonals):

```
| a b c d |
| e a b c |
| f e a b |
| g f e a |
```
The Toeplitz matrix contains the predicted data trace in each column such taht it represents the temporal convolution of $\bf p$ with $\bf w$. Here, $\bf w$ is a simple acausal (depending on past, present, and future data) 1D Weiner filter. Note that the coefficients are a function of the data and thus the models. The filter is designed not to match individual events in $\bf p$ separately to to an equivalent event in $\bf d$, but instance match the entire predicted trace to an observed one. The AWI method is based directly on matching entire observed wavefields.

The $g$ equation represents a linear least-squares problem with a single global minimum and no secondary local minima so that cycle skipping is not an issue. Its solution is given by

$$\mathbf{w} = (\mathbf{P}^T \mathbf{P})^{-1} \mathbf{P}^T \mathbf{d}.$$

Note here any practical method would add a small positive number to the diagonal of the matrix $\bf P^T P$ to stabilise.

#### Seismic inversion

For a convolutional Wiener filter, the identity filter is a unit amplitude delta at zero lag. The seismic inversion objective function is therefore set up to minimise an objective function $f$ where

$$f = \frac 1 2 \frac {\| \mathbf{Tw} \|^2} {\| \mathbf{w} \|^2}.$$

$\bf T$ is a diagonal matrix which weights the coefficients of $\bf w$ by some continuous monotonic functions of the magnitude of their temporal lag. There are many forms $\bf T$ can take, but AWI does not appear to be sensitive to changing forms. This objective penalises Wiener cofficients at large lags, and the penalty increases as the lag does. As the method is iterated, the objective function forces $\bf w$ towards a delta function at zero lag. This is the most conceptually straightforward representation, but not the most mathematically elegant. It is refered to as forward AWI.

A simpler outcome is obtains by finding a filter $\bf v$ whcih transforms the data into the predicted data by solving

$$\bf v = (D^T D)^{-1} D^T p,$$

and minimising

$$f = \frac 1 2 \frac {\| \mathbf{Tv} \|^2 } { \| v \|^2 },$$

where $\bf D$ represents convolution by the observed data. A more thorough explanation of $D$ is in the [implications section](#implications) Here, the two filters $\bf w$ and $\bf v$ are least-squares convolutional inverses, meaning if they are convolved then the result will be a band-limited unit-amplitude delta. This version is caled reverse AWI.

## Gradient and adjoint source

In traditional FWI, the expression for the gradient of the functional w.r.t the model parameters can be written

$$\frac {\partial f}{\partial \mathbf{m}} = -\mathbf{u}^T \left( \frac {\partial \mathbf{A}} {\partial \mathbf{m}} \right)^T \mathbf{A}^{-T}\mathbf{R}^T\delta \mathbf{s},$$

where $\bf R$ is a restriction matrix, and $\delta \bf s$ is the adjoint source. To compute the gradient we thus must find the adjoint source.

The adjoint source is given by the gradient of the objective w.r.t the predicted data:

$$\delta \mathbf{s} = \frac {\partial \mathbf{x}^T} {\partial \mathbf{p}}\mathbf{x} = \frac {\partial f} {\partial \mathbf{p}}.$$
In traditional FWI, this is simply the data residual. In AWI, however, the adjoint source is more complex.

Using the forward AWI definition of the matching filter, the adjoint source is given by

$$\partial \mathbf{s} = \mathbf{W^T P (P^T P)^{-1}} \left( \frac {2f\mathbf{I} - \mathbf{T}^2} {\mathbf{w^T w}} \right)\mathbf{w}.$$

Here, $\mathbf{w}^T$ represents the cross correlation between the filter $\bf w$. From right to left, this reads:

- find the Wiener filter which matches the predicted data to the observed data in a least squares sense
- normalise this by the inner product,
- weight the coefficients using the function represented by the matrix $(2 f \mathbf{i} - \mathbf{T}^2)$ that depends on the lag of each coeff and the current value of the functional,
- deconvolve this by the auto correlation $\bf P^T P$ of the predicted data,
- convolve the result with the predicted data $\bf P$,
- and finally crosscorrelate with the original Wiener filter.

If we instead use the reverse AWI definition, we find

$$\partial \mathbf{s} = \mathbf{D (D^T D)^{-1}} \left( \frac {\mathbf{T}^2 - 2f\mathbf{I}} {\mathbf{v^T v}} \right)\mathbf{v}.$$

### AWI method summary

In the simplest terms, AWI consists of computing a forward wavefield for each physical source from which the predicted data can be extracted. A suite of 1D acausal Wiener filters can then be generated, each of which transforms the data into its observed trace or vice versa. These filters are manipulated using one of the adjoint source equations. From there, the usual FWI routine continues. Basically the one different step is that the adjoint source is computed differently. The matching filter is essentially a more sophisticated way of asking "how wrong is my model?" compared to FWI's blunt amplitude subtraction.

#### Implications

This is nice, because we need not formulate a new operator or anything like that. In fact, we can write an entire FWI program and just drop in a different residual to achieve everything that has been explained above.

In practice, we need not actually compute all the filters. Remember the filter solution can be written as

$$\bf v = (D^T D)^{-1} D^T p,$$

$\bf D$ is a convolution matrix. More specifically, it represents the convolution of the filter with the predicted data. It is formed by stacking the time-shifted copies of the predicted data as columns or rows:

$$\begin{bmatrix}
d_0 & 0 & 0 \\
d_1 & d_0 & 0 \\
d_2 & d_1 & d_0 \\
\vdots & \vdots & \vdots \\
d_n & d_{n-1} & d_{n-2}\\
0 & d_n & d_{n-1}\\
0 & 0 & d_n\\
\end{bmatrix},$$

such that $\bf D v = p \ast v \approx d$.

With this representation of $\bf v,$ we need never actually form it. Instead of solving the $\bf v$ equation explicitly (an expenive inverse solve), all quantities can be computed by matrix-vector products using $\bf D$.
