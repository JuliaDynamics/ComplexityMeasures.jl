export NaiveKernel
using Neighborhood: Theiler, KDTree, bulkisearch, searchstructure
using Distances: Metric, Euclidean

"""
    NaiveKernel(ϵ::Real, ss = KDTree; w = 0, metric = Euclidean()) <: ProbabilitiesEstimator

Estimate probabilities/entropy using a "naive" kernel density estimation approach (KDE), as 
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

Probabilities ``P(\\mathbf{x}, \\epsilon)`` are assigned to every point ``\\mathbf{x}`` by 
counting how many other points occupy the space spanned by 
a hypersphere of radius `ϵ` around ``\\mathbf{x}``, according to:

```math
P_i( X, \\epsilon) \\approx \\dfrac{1}{N} \\sum_{s \\neq i } B(||X_i - X_j|| < \\epsilon),
```

where ``B`` gives 1 if the argument is `true`. Probabilities are then normalized.

The search structure `ss` is any search structure supported by Neighborhood.jl.

The keyword `w` stands for the [Theiler window](@ref), and excludes indices ``s``
that are within ``|i - s| ≤ w`` from the given point ``\\mathbf{x}_i``.

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
