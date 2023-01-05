import SpecialFunctions: digamma, gamma
using Neighborhood: Theiler, KDTree, bulksearch

function maximum_neighbor_distances(pts, w::Int, k::Int)
    theiler = Theiler(w)
    w >= 0 || error("w, the number of neighbors to exclude, must be >= 0")
    tree = KDTree(pts) # Euclidean metric forced
    idxs, dists = bulksearch(tree, pts.data, NeighborNumber(k), theiler)
    # Distance to k-th nearest neighbor for each of the points
    return [d[end] for d in dists]
end

"Volume of a unit ball in R^d."
ball_volume(d::Int) = π^(d/2)/gamma((d/2)+1)

include("KozachenkoLeonenko.jl")
include("Kraskov.jl")
include("Zhu.jl")
include("ZhuSingh.jl")
include("Gao.jl")
include("Goria.jl")
include("Lord.jl")
