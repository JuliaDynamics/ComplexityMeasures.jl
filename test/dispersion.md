# Dispersion probabilities and entropies

## Dispersion entropy

### Symbolizing using cumulative distribution functions

Assume we have a univariate time series $X = \{x_i\}_{i=1}^N$. The dispersion entropy algorithm first maps each $x_i$ to a new real number $y_i \in [0, 1]$ by using the normal cumulative distribution function (CDF), $x_i \to y_i : y_i = \int_{-\infty}^{x_i} e^{(-(x_i - \mu)^2)/(2\sigma^2)}$, where $\mu$ and $\sigma$ are the empirical mean and standard deviation of $X$. Other choices of CDFs are also possible, but in Entropies.jl currently only implements the normal CDF ([`GaussianSymbolization`](@ref)), which was used in the original dispersion entropy paper.

Next, each $y_i$ is linearly mapped to an integer $z_i \in [1, 2, \ldots, c]$ using the map $y_i \to z_i : z_i = R(y_i(c-1) + 0.5)$, where $c$ is the number of categories and $R$ indicates rounding up to the nearest integer.

This procedure subdivides the interval $[0, 1]$ into a set of subintervals that form a covering of $[0, 1]$, and assigns each $y_i$ to one of these subintervals. The original time series $X$ is thus transformed to a symbol time series $S = \{s_i\}_{i=1}^N$, where $s_i \in [1, 2, \ldots, c]$.

### Dispersion patterns

In the next step, the symbol time series $S$ is embedded into an $m$-dimensional integer-valued time series, using an embedding lag of $\tau = 1$, which yields a total of $N - (m - 1)\tau$ points. Because each $z_i$ can take on $c$ different values, and each embedding point has $m$ values, there are $c^m$ possible values that each embedding point can take.

Each embedding vector is called a "dispersion pattern". For illustration, let's consider the case when $m = 5$, $c = 3$ and use some very imprecise terminology:

When $c = 3$, the "outliers" below the mean are in one group, values close to the mean are in one group, and "outliers" above the mean are in a third group. Then the embedding vector $[2, 2, 2, 2, 2]$ consists of values that are relatively close together (close to the mean), so it represents a set of numbers that are not very spread out (less dispersed). The embedding vector $[1, 1, 2, 3, 3]$, however, represents numbers that are much more spread out (more dispersed), because the categories representing "outliers" both above and below the mean are represented, not only values close to the mean.

A probability distribution $P = \{p_i \}_{i=1}^{c^m}$, where $\sum_i^{c^m} p_i = 1$, can then be estimated by counting and sum-normalising the distribution of dispersion patterns among the embedding vectors. In Entropies.jl, the entire procedure above is performed by the [`Dispersion`](@ref) probabilities estimator, i.e.

```@example dispersion_entropy
using Entropies
x = rand(1000);
c, m = 3, 5
est = Dispersion(s = GaussianSymbolization(c), m = m)
probs = probabilities(x, est)
```

Dispersion entropy is then computed by feeding these probabilites into the formula for generalized Renyi entropy with order `q = 1`, e.g.

```@example dispersion_entropy
entropy_renyi(probs, q = 1, base = MathConstants.e)
```

Any order `q` can be used if computing non-normalized dispersion entropy. For normalized dispersion entropy, only `q == 1` is valid.

## Reverse dispersion entropy

Li et al. (2021) defines the reverse dispersion entropy as

```math
H_{rde} = \sum_{i = 1}^{c^m} \left(p_i - \dfrac{1}{{c^m}} \right)^2.
```

where the probabilities $p_i$ are obtained precisely as for the dispersion entropy.

The minimum value of $H_{rde}$ occurs precisely when the probability distribution is flat, which occurs when all $p_i$s are equal to $1/c^m$. $H_{rde}$ can therefore be said to be a measure of how far the dispersion pattern probability distribution is from white noise.

## A clarification on notation

With ambiguous notation, Li et al. claim that

$H_{rde} = \sum_{i = 1}^{c^m} \left(p_i - \dfrac{1}{{c^m}} \right)^2 = \sum_{i = 1}^{c^m} p_i^2 - \frac{1}{c^m}$.

But on the right-hand side of the equality, does the constant term appear within or outside the sum?

Let's see. Using (in step 4) that $P$ is a probability distribution by construction, we see that the constant must appear *outside* the sum:

```math
\begin{align}
H_{rde} &= \sum_{i = 1}^{c^m} \left(p_i - \dfrac{1}{{c^m}} \right)^2 
= \sum_{i=1}^{c^m} p_i^2 - \frac{2p_i}{c^m} + \frac{1}{c^{2m}} \\
&= \left( \sum_{i=1}^{c^m} p_i^2 \right) - \left(\sum_i^{c^m} \frac{2p_i}{c^m}\right) + \left( \sum_{i=1}^{c^m} \dfrac{1}{{c^{2m}}} \right) \\
&= \left( \sum_{i=1}^{c^m} p_i^2 \right) - \left(\frac{2}{c^m} \sum_{i=1}^{c^m} p_i \right) +  \dfrac{c^m}{c^{2m}} \\
&= \left( \sum_{i=1}^{c^m} p_i^2 \right) - \frac{2}{c^m} (1) +  \dfrac{1}{c^{m}} \\
&= \left( \sum_{i=1}^{c^m} p_i^2 \right) - \dfrac{1}{c^{m}}. \\
\end{align}
```
