export NaiveKernel, TreeDistance, DirectDistance, probabilities

import Distances: Metric, Euclidean, evaluate
import NearestNeighbors: NNTree, KDTree, inrange
import DelayEmbeddings: AbstractDataset

""" Abstract type for different estimation methods of kernel densities """
abstract type KernelEstimationMethod end

"""
    DirectDistance(metric::M = Euclidean()) <: KernelEstimationMethod

Pairwise distances are evaluated directly using the provided metric.
"""
struct DirectDistance{M<:Metric} <: KernelEstimationMethod
    metric::M
    function DirectDistance(metric::M = Euclidean()) where M
        new{M}(metric)
    end
end

"""
    TreeDistance(metric::M = Euclidean()) <: KernelEstimationMethod

Pairwise distances are evaluated using a tree-based approach with the provided `metric`.
"""
struct TreeDistance{M<:Metric} <: KernelEstimationMethod
    metric::M
    function TreeDistance(metric::M = Euclidean()) where M
        new{M}(metric)
    end
end

"""
    NaiveKernel(ϵ::Real, method::KernelEstimationMethod = TreeDistance()) <: ProbabilitiesEstimator

Estimate probabilities/entropy using a "naive" kernel density estimation approach (KDE), as 
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

Probabilities ``P(\\mathbf{x}, \\epsilon)`` are assigned to every point ``\\mathbf{x}`` by 
counting how many other points occupy the space spanned by 
a hypersphere of radius `ϵ` around ``\\mathbf{x}``, according to:

```math
P_i( \\mathbf{x}, \\epsilon) \\approx \\dfrac{1}{N} \\sum_{s \\neq i } K\\left( \\dfrac{||\\mathbf{x}_i - \\mathbf{x}_s ||}{\\epsilon} \\right),
```

where ``K(z) = 1`` if ``z < 1`` and zero otherwise. Probabilities are then normalized.

## Methods 

- Tree-based evaluation of distances using [`TreeDistance`](@ref). Faster, but more
    memory allocation.
- Direct evaluation of distances using [`DirectDistance`](@ref). Slower, but less 
    memory allocation. Also works for complex numbers.

## Estimation

Probabilities or entropies can be estimated from `Dataset`s.

- `probabilities(x::AbstractDataset, est::NaiveKernel)`. Associates a probability `p` to 
    each point in `x`.
- `genentropy(x::AbstractDataset, est::NaiveKernel)`.  Associate probability `p` to each 
    point in `x`, then compute the generalized entropy from those probabilities.

## Examples

```julia
using Entropy, DelayEmbeddings
pts = Dataset([rand(5) for i = 1:10000]);
ϵ = 0.2
est_direct = NaiveKernel(ϵ, DirectDistance())
est_tree = NaiveKernel(ϵ, TreeDistance())

p_direct = probabilities(pts, est_direct)
p_tree = probabilities(pts, est_tree)

# Check that both methods give the same probabilities
all(p_direct .== p_tree)
```

See also: [`DirectDistance`](@ref), [`TreeDistance`](@ref).

[^PrichardTheiler1995]: Prichard, D., & Theiler, J. (1995). Generalized redundancies for time series analysis. Physica D: Nonlinear Phenomena, 84(3-4), 476-493.
"""
struct NaiveKernel{KM<:KernelEstimationMethod} <: ProbabilitiesEstimator
    ϵ::Real
    method::KM
    
    function NaiveKernel(ϵ::Real, method::KM = TreeDistance()) where KM <: KernelEstimationMethod
        ϵ > 0 || error("Radius ϵ must be larger than zero, otherwise no radius around the points is defined!")
        new{KM}(ϵ, method)
    end
end

function tree_distance!(p, tree, x, ϵ, sort::Bool, N)
    # There's no need to sort the indices, we just count the number of points within radius ϵ.    
    idxs = inrange(tree, x, ϵ, false)
    
    @inbounds for i = 1:N
        p[i] = length(idxs[i])
    end

    return p
end

function get_pts_within_radius(x::AbstractDataset{D, T}, est::NaiveKernel{<:M}) where {D, T, M<:TreeDistance}
    N = length(x)
    tree = KDTree(x, est.method.metric)
    p = zeros(T, N)

    # Count number of points within radius of ϵ for each point in `x`
    tree_distance!(p, tree, x.data, est.ϵ, false, N)
    
    return p
end

function direct_distance!(p, x, metric, ϵ, N) where {D, T}
    @inbounds for i in 1:N
        p[i] = count(evaluate(metric, x[i], x[j]) < ϵ for j in 1:N)
    end

    return p
end

function get_pts_within_radius(x::AbstractDataset{D, T}, est::NaiveKernel{M}) where {D, T, M<:DirectDistance}
    N = length(x)
    p = zeros(T, N)
    direct_distance!(p, x, est.method.metric, est.ϵ, N)
    return p
end


function probabilities(x::Dataset, est::NaiveKernel)
    N = length(x)
    p = get_pts_within_radius(x, est)
    return Probabilities(p ./= sum(p))
end
