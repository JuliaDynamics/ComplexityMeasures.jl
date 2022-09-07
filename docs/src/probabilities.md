# [Probabilities](@id estimators)

For categorical or integer-valued data, probabilities can be estimated by directly counting relative frequencies of data elements. For such data, use `probabilities(x::Array_or_Dataset) → p::Probabilities`.

More advanced estimators computing probabilities by first either discretizing, symbolizing or transforming the data in a way that quantifies some useful properties about the underlying data (e.g. visitation frequencies, wavelet energies, or permutation patterns), from which probability distributions can be estimated. Use `probabilities(x::Array_or_Dataset, est::ProbabilitiesEstimator)` in combination with any of the estimators listed below.

```@docs
Probabilities
probabilities
probabilities!
ProbabilitiesEstimator
```

## Estimators

### Count occurrences (counting)

```@docs
CountOccurrences
```

### Permutation (symbolic)

```@docs
SymbolicPermutation
SpatialSymbolicPermutation
```

### Visitation frequency (binning)

```@docs
VisitationFrequency
```

#### Specifying binning/boxes

```@docs
RectangularBinning
```

### Transfer operator (binning)

```@docs
TransferOperator
```

#### Utility methods/types

```@docs
InvariantMeasure
invariantmeasure
transfermatrix
```

### Kernel density

```@docs
NaiveKernel
```

#### Example

Here, we draw some random points from a 2D normal distribution. Then, we use kernel 
density estimation to associate a probability to each point `p`, measured by how many 
points are within radius `1.5` of `p`. Plotting the actual points, along with their 
associated probabilities estimated by the KDE procedure, we get the following surface 
plot.

```@example MAIN
using DynamicalSystems, CairoMakie, Distributions
𝒩 = MvNormal([1, -4], 2)
N = 500
D = Dataset(sort([rand(𝒩) for i = 1:N]))
x, y = columns(D)
p = probabilities(D, NaiveKernel(1.5))
fig, ax = surface(x, y, p.p; axis=(type=Axis3,))
ax.zlabel = "P"
ax.zticklabelsvisible = false
fig
```

### Wavelet

```@docs
WaveletOverlap
```
