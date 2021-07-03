export NaiveKernel, TreeDistance, DirectDistance, probabilities

import Distances: Metric, Euclidean, evaluate
import NearestNeighbors: KDTree, inrange
import DelayEmbeddings: AbstractDataset

""" Abstract type for different estimation methods of kernel densities """
abstract type KernelEstimationMethod end

"""
    DirectDistance(metric = Euclidean()) <: KernelEstimationMethod

Pairwise distances are evaluated directly using the provided `metric`.
"""
struct DirectDistance{M<:Metric} <: KernelEstimationMethod
    metric::M
end
DirectDistance() = DirectDistance(Euclidean())

"""
    TreeDistance(metric = Euclidean()) <: KernelEstimationMethod

Pairwise distances are evaluated using a KDTree with the provided `metric`.
"""
struct TreeDistance{M<:Metric} <: KernelEstimationMethod
    metric::M
end
TreeDistance() = TreeDistance(Euclidean())

"""
    NaiveKernel(ϵ::Real, ss = KDTree; w = 0) <: ProbabilitiesEstimator

Estimate probabilities/entropy using a "naive" kernel density estimation approach (KDE), as 
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

Probabilities ``P(\\mathbf{x}, \\epsilon)`` are assigned to every point ``\\mathbf{x}`` by 
counting how many other points occupy the space spanned by 
a hypersphere of radius `ϵ` around ``\\mathbf{x}``, according to:

```math
P_i( \\mathbf{x}, \\epsilon) \\approx \\dfrac{1}{N} \\sum_{s \\neq i } K\\left( \\dfrac{||\\mathbf{x}_i - \\mathbf{x}_s ||}{\\epsilon} \\right),
```

where ``K(z) = 1`` if ``z < 1`` and zero otherwise. Probabilities are then normalized.

The search structure `ss` comes from Neighborhood.jl and can be either: 
- `KDTree`: Tree-based evaluation of distances. Faster, but more memory allocation.
- `BruteForce`: Direct evaluation of all distances. Extremely slow.

The keyword `w` stands for the [Theiler window](@ref), and excludes indices ``s``
that are within ``|i - s| ≤ w`` from the given point ``\\mathbf{x}_i``.

[^PrichardTheiler1995]: Prichard, D., & Theiler, J. (1995). Generalized redundancies for time series analysis. Physica D: Nonlinear Phenomena, 84(3-4), 476-493.
"""
struct NaiveKernel{KM<:KernelEstimationMethod} <: ProbabilitiesEstimator
    ϵ::Float64
    method::KM
    function NaiveKernel(ϵ::Real, method::KM = TreeDistance()) where KM <: KernelEstimationMethod
        ϵ > 0 || error("Radius ϵ must be larger than zero!")
        new{KM}(ϵ, method)
    end
end

function probabilities(x::AbstractDataset, est::NaiveKernel)
    p = get_pts_within_radius(x, est)
    return Probabilities(p ./= sum(p))
end

function get_pts_within_radius(x::AbstractDataset, est::NaiveKernel{<:TreeDistance})
    N = length(x)
    tree = KDTree(x, est.method.metric)
    p = zeros(eltype(x), N)

    # Count number of points within radius of ϵ for each point in `x`
    tree_distance!(p, tree, x.data, est.ϵ, false, N)
    
    return p
end
function get_pts_within_radius(x::AbstractDataset{D, T}, est::NaiveKernel{M}) where {D, T, M<:DirectDistance}
    N = length(x)
    p = zeros(T, N)
    direct_distance!(p, x, est.method.metric, est.ϵ, N)
    return p
end

function tree_distance!(p, tree, x, ϵ, sort::Bool, N)
    # There's no need to sort the indices, we just count the number of points within radius ϵ.    
    idxs = inrange(tree, x, ϵ, false)
    
    @inbounds for i = 1:N
        p[i] = length(idxs[i])
    end

    return p
end


function direct_distance!(p, x, metric, ϵ, N) where {D, T}
    @inbounds for i in 1:N
        p[i] = count(evaluate(metric, x[i], x[j]) < ϵ for j in 1:N)
    end

    return p
end




