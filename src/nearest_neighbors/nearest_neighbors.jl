import SpecialFunctions: digamma, gamma
using NearestNeighbors


abstract type NearestNeighborEntropyEstimator <: EntropyEstimator end


""" Assumes the estimator has a `k` field incidating the number of nearest neighbors to consider and a `w` field 
indicating the number of neighbors to skip 
"""
function get_ρs(pts, est::EntropyEstimator)
    est.w >= 0 || error("w, the number of neighbors to exclude, must be >= 0")
    tree = KDTree(pts)
    idxs, dists = knn(tree, pts.data, est.k + est.w + 1, true)

    # Distance to k-th nearest neighbor for each of the points (considering also exlusion radius)
    return [d[end] for d in dists]
end

""" Volume of a unit ball in R^d. """
V(d::Int) = π^(d/2)/gamma((d/2)+1)

include("KozachenkoLeonenko.jl")
include("Kraskov.jl")