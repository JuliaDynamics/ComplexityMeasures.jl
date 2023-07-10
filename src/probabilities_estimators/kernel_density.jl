using Neighborhood: Theiler, KDTree, BruteForce, bulkisearch, searchstructure
using Distances: Metric, Euclidean
export NaiveKernel, KDTree, BruteForce

"""
    NaiveKernel(ϵ::Real; method = KDTree, w = 0, metric = Euclidean()) <: ProbabilitiesEstimator

Estimate probabilities/entropy using a "naive" kernel density estimation approach (KDE), as
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

Probabilities ``P(\\mathbf{x}, \\epsilon)`` are assigned to every point ``\\mathbf{x}`` by
counting how many other points occupy the space spanned by
a hypersphere of radius `ϵ` around ``\\mathbf{x}``, according to:

```math
P_i( X, \\epsilon) \\approx \\dfrac{1}{N} \\sum_{s} B(||X_i - X_j|| < \\epsilon),
```

where ``B`` gives 1 if the argument is `true`. Probabilities are then normalized.

## Keyword arguments

- `method = KDTree`: the search structure supported by Neighborhood.jl.
  Specifically, use `KDTree` to use a tree-based neighbor search, or `BruteForce` for
  the direct distances between all points. KDTrees heavily outperform direct distances
  when the dimensionality of the data is much smaller than the data length.
- `w = 0`: the Theiler window, which excludes indices ``s`` that are within
  ``|i - s| ≤ w`` from the given point ``x_i``.
- `metric = Euclidean()`: the distance metric.

## Outcome space

The outcome space `Ω` for `NaiveKernel` are the indices of the input data, `eachindex(x)`.
Hence, input `x` is needed for a well-defined [`outcome_space`](@ref).
The reason to not return the data points themselves is because duplicate data points may
not get assigned same probabilities (due to having different neighbors).

[^PrichardTheiler1995]:
    Prichard, D., & Theiler, J. (1995). Generalized redundancies for time series analysis.
    Physica D: Nonlinear Phenomena, 84(3-4), 476-493.
"""
struct NaiveKernel{KM, M <: Metric} <: ProbabilitiesEstimator
    ϵ::Float64
    method::KM
    w::Int
    metric::M
end
function NaiveKernel(ϵ::Real; method = KDTree, w = 0, metric = Euclidean())
    ϵ ≤ 0 && throw(ArgumentError("Radius ϵ must be larger than zero!"))
    return NaiveKernel(ϵ, method, w, metric)
end

function probabilities_and_outcomes(est::NaiveKernel, x::AbstractStateSpaceSet)
    theiler = Theiler(est.w)
    ss = searchstructure(est.method, vec(x), est.metric)
    idxs = bulkisearch(ss, vec(x), WithinRange(est.ϵ), theiler)
    p = Float64.(length.(idxs))
    return Probabilities(p), eachindex(x)
end

outcome_space(::NaiveKernel, x) = eachindex(x)
