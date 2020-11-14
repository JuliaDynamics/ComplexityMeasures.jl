import Distances: Metric, Euclidean, evaluate
import NearestNeighbors: NNTree, KDTree, inrange

""" Abstract type for different estimation methods of kernel densities """
abstract type KernelEstimationMethod end

"""
    DirectDistance(metric::M = Euclidean()) <: KernelEstimationMethod

Estimate kernel densities by direct evaluation of pairwise distances.
"""
struct DirectDistance{M<:Metric} <: KernelEstimationMethod
    metric::M
    function DirectDistance(metric::M = Euclidean()) where M
        new{M}(metric)
    end
end

"""
    Tree(metric::M = Euclidean()) <: KernelEstimationMethod

Estimate kernel densities by a tree-based method of computing pairwise distances.
"""
struct Tree{M<:Metric} <: KernelEstimationMethod
    metric::M
    function Tree(metric::M = Euclidean()) where M
        new{M}(metric)
    end
end


"""
    NaiveKernel(ϵ::Real) <: ProbabilitiesEstimator

Estimate probabilities/entropy using a kernel density estimation (KDE). This is the "naive kernel" approach 
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

Probabilities ``P(\\mathbf{x}, \epsilon)`` are assigned to every point ``\\mathbf{x}`` by counting how many other points occupy the space spanned by 
a hypersphere of radius `ϵ` around ``\\mathbf{x}`` (and normalized after), according to:

```math
P_i(\\mathbf{x}, \\epsilon) \\approx \\dfrac{1}{N} \\sum_{s \\neq i } K\\left(\\dfrac{||\\mathbf{x}_i - \\mathbf{x}_s ||}{\\epsilon} {\right)
```,

where ``K(z) = 1`` if ``z < 1`` and zero otherwise.

[^PrichardTheiler1995]: Prichard, D., & Theiler, J. (1995). Generalized redundancies for time series analysis. Physica D: Nonlinear Phenomena, 84(3-4), 476-493.
"""
struct NaiveKernel{KM<:KernelEstimationMethod} <: ProbabilitiesEstimator
    ϵ::Real
    method::KM
    function NaiveKernel(ϵ::Real; method::KM = Tree()) where KM
        ϵ > 0 || error("Radius ϵ must be larger than zero, otherwise no radius around the points is defined!")
        new{KM}(ϵ, method)
    end
end

function get_pts_within_radius(pts, est::NaiveKernel{<:Tree})
    tree = KDTree(pts, est.method.metric)

    # Count number of points within radius of ϵ for each point in `pts`
    # There's no need to sort the indices, we just count the number of points within radius ϵ.
    idxs = inrange(tree, pts.data, est.ϵ, false) 
    return [length(idx) for idx in idxs] # if we're actually getting the points and normalizing later, no need to do ./ N here
end

function get_pts_within_radius(pts, est::NaiveKernel{<:DirectDistance})
    N = length(pts)
    p = zeros(eltype(pts), N)
    @inbounds for i in 1:N
        p[i] = count(evaluate(est.method.metric, pts[i], pts[j]) < est.ϵ for j in 1:N)
    end

    return p
end

function probabilities(x::Dataset, est::NaiveKernel)
    N = length(x)
    p = get_pts_within_radius(x, est)
    return p ./ sum(p)
end