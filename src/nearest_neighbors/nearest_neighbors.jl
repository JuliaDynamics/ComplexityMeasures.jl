import SpecialFunctions: digamma, gamma
using NearestNeighbors


abstract type NearestNeighborEntropyEstimator <: EntropyEstimator end

function maximum_neighbor_distances(pts, est::EntropyEstimator)
    est.w >= 0 || error("w, the number of neighbors to exclude, must be >= 0")
    tree = KDTree(pts)
    idxs, dists = knn(tree, pts.data, est.k + est.w + 1, true)

    # Distance to k-th nearest neighbor for each of the points (considering also exlusion radius)
    return [d[end] for d in dists]
end

""" Volume of a unit ball in R^d. """
ball_volume(d::Int) = π^(d/2)/gamma((d/2)+1)

include("KozachenkoLeonenko.jl")
include("Kraskov.jl")