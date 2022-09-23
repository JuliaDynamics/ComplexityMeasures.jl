using Neighborhood: Theiler, KDTree, BruteForce, bulkisearch, searchstructure
using Distances: Metric, Euclidean
export NaiveKernel, KDTree, BruteForce
export entropy_kernel
"""
    NaiveKernel(ϵ::Real, ss = KDTree; w = 0, metric = Euclidean()) <: ProbabilitiesEstimator

Estimate probabilities/entropy using a "naive" kernel density estimation approach (KDE), as
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

Probabilities ``P(\\mathbf{x}, \\epsilon)`` are assigned to every point ``\\mathbf{x}`` by
counting how many other points occupy the space spanned by
a hypersphere of radius `ϵ` around ``\\mathbf{x}``, according to:

```math
P_i( X, \\epsilon) \\approx \\dfrac{1}{N} \\sum_{s} B(||X_i - X_j|| < \\epsilon),
```

where ``B`` gives 1 if the argument is `true`. Probabilities are then normalized.

The search structure `ss` is any search structure supported by Neighborhood.jl.
Specifically, use `KDTree` to use a tree-based neighbor search, or `BruteForce` for
the direct distances between all points. KDTrees heavily outperform direct distances
when the dimensionality of the data is much smaller than the data length.

The keyword `w` stands for the [Theiler window](@ref), and excludes indices ``s``
that are within ``|i - s| ≤ w`` from the given point ``X_i``.

[^PrichardTheiler1995]: Prichard, D., & Theiler, J. (1995). Generalized redundancies for time series analysis. Physica D: Nonlinear Phenomena, 84(3-4), 476-493.
"""
struct NaiveKernel{KM, M <: Metric} <: ProbabilitiesEstimator
    ϵ::Float64
    method::KM
    w::Int
    metric::M
end
function NaiveKernel(ϵ::Real, method = KDTree; w = 0, metric = Euclidean())
    ϵ > 0 || error("Radius ϵ must be larger than zero!")
    return NaiveKernel(ϵ, method, w, metric)
end

function probabilities(x::DelayEmbeddings.AbstractDataset, est::NaiveKernel)
    theiler = Theiler(est.w)
    ss = searchstructure(est.method, x.data, est.metric)
    idxs = bulkisearch(ss, x.data, WithinRange(est.ϵ), theiler)
    p = Float64.(length.(idxs))
    return Probabilities(p)
end

"""
    entropy_kernel(x; ϵ::Real = 0.2*StatsBase.std(x), method = KDTree; w = 0,
        metric = Euclidean(), base = MathConstants.e)

Calculate Shannon entropy using the "naive" kernel density estimation approach (KDE), as
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

Shorthand for `entropy_renyi(x, NaiveKernel(ϵ, method, w = w, metric = metric), q = 1, base = base)`.

See also: [`NaiveKernel`](@ref), [`entropy_renyi`](@ref).

[^PrichardTheiler1995]: Prichard, D., & Theiler, J. (1995). Generalized redundancies for time series analysis. Physica D: Nonlinear Phenomena, 84(3-4), 476-493.
"""
function entropy_kernel(x; ϵ::Real = 0.2*StatsBase.std(x), method = KDTree, w = 0,
        metric = Euclidean(), base = MathConstants.e)
    est = NaiveKernel(ϵ, method, w = w, metric = metric)
    return entropy_renyi(x, est; base = base, q = 1)
end