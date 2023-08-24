using StaticArrays: @MVector, @MMatrix, @SVector, SVector, MVector
using Neighborhood: NeighborNumber, KDTree, Theiler, Euclidean
using Neighborhood: bulksearch
using LinearAlgebra: svd

export Lord


############################################################################################
# Some benchmarking.
# This is for a 10000x2 dataset with k = 30 neighbors.
############################################################################################
#    1. (135.538 ms (380046 allocations: 55.42 MiB) # total)
#    2.  24.038 ms (50040 allocations: 7.82 MiB) # neighbor searches
#    3. 25.484 ms (60040 allocations: 13.00 MiB) # finding the neighbors neighborsᵢ
#    4. 123.084 ms (230040 allocations: 52.07 MiB) # svd, can't optimize this
#    5. 135.538 ms (380046 allocations: 55.42 MiB) # everything after svd
#
# This is pretty optimized now, but there is still room for significant improvements
# in the svd step. However, there isn't a non-allocating `svd` in base Julia, or in
# StaticArrays at the moment, so we'd have to implement this ourselves.
# It is not even clear if a nonallocating version would be faster.
############################################################################################

"""
    Lord <: DifferentialInfoEstimator
    Lord(measure = Shannon(); k = 10, w = 0)

The `Lord` estimator (Lord et al., 2018)[Lord2018](@cite) estimates the [`Shannon`](@ref)
differential [`information`](@ref) using a nearest neighbor approach with a local
nonuniformity correction (LNC), with logarithms to the `base` specified in `definition`.

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

## Description

Assume we have samples ``\\bar{X} = \\{\\bf{x}_1, \\bf{x}_2, \\ldots, \\bf{x}_N \\}`` from a
continuous random variable ``X \\in \\mathbb{R}^d`` with support ``\\mathcal{X}`` and
density function ``f : \\mathbb{R}^d \\to \\mathbb{R}``. `Lord` estimates the
[Shannon](@ref) differential entropy

```math
H(X) = \\int_{\\mathcal{X}} f(x) \\log f(x) dx = \\mathbb{E}[-\\log(f(X))],
```

by using the resubstitution formula

```math
\\hat{\\bar{X}, k} = -\\mathbb{E}[\\log(f(X))]
\\approx \\sum_{i = 1}^N \\log(\\hat{f}(\\bf{x}_i)),
```

where ``\\hat{f}(\\bf{x}_i)`` is an estimate of the density at ``\\bf{x}_i`` constructed
in a manner such that ``\\hat{f}(\\bf{x}_i) \\propto \\dfrac{k(x_i) / N}{V_i}``,
where ``k(x_i)`` is the number of points in the neighborhood of ``\\bf{x}_i``, and ``V_i``
is the volume of that neighborhood.

While most nearest-neighbor based differential entropy estimators uses regular volume
elements (e.g. hypercubes, hyperrectangles, hyperspheres) for approximating the
local densities ``\\hat{f}(\\bf{x}_i)``, the `Lord` estimator uses hyperellopsoid volume
elements. These hyperellipsoids are, for each query point `xᵢ`, estimated using singular
value decomposition (SVD) on the `k`-th nearest neighbors of `xᵢ`. Thus, the hyperellipsoids
stretch/compress in response to the local geometry around each sample point. This
makes `Lord` a well-suited entropy estimator for a wide range of systems.
"""
struct Lord{I <: InformationMeasure} <: NNDifferentialInfoEstimator{I}
    definition::I
    k::Int
    w::Int
end
function Lord(definition = Shannon(); k = 10, w = 0)
    return Lord(definition, k, w)
end

function information(est::Lord{<:Shannon}, x::AbstractStateSpaceSet{D}) where {D}
    (; k, w) = est
    N = length(x)
    tree = KDTree(x, Euclidean())
    knn_idxs, ds = bulksearch(tree, x, NeighborNumber(k), Theiler(w))

    # Decrease allocations and speed up computations by pre-allocating.
    # We're only dealing with matrices which in either axis has a maximum dimension of `k`,
    # so this is still far within the realm of where StaticArrays shines.
    # -------------------------------------------------------------------------------------
    rs = @MVector zeros(D) # Scaled ellipsoid axis lengths
    Λ = @MMatrix zeros(D, D) # Hyperellipsoid matrix

    # C contains neighborhood-centroid-centered vectors, where
    # `C[1]` := the centered query point
    # `C[1 + j]` := the centered `j`-th neighbor of the query point.
    C = [@SVector zeros(D) for i = 1:k+1]

    # Centered neighbors need to be ordered row-wise in a matrix. We re-fill this matrix
    # for every query point `xᵢ`
    A = @MMatrix zeros(k+1, D)

    # Precompute some factors
    γ = gamma(1 + D/2)
    f = N * π^(D/2)

    h = 0.0
    for (i, xᵢ) in enumerate(x)
        neighborsᵢ = @views x[knn_idxs[i]]
        # Center neighborhood points around mean of the neighborhood.
        c = centroid(xᵢ, neighborsᵢ, k)
        center_neighborhood!(C, c, xᵢ, neighborsᵢ) # put centered vectors in `C`
        fill_A!(A, C)

        # SVD. The columns of Vt are the semi-axes of the ellipsoid, while Σ gives the
        # magnitudes of the axes.
        U, Σ, Vt = svd(A) # it is actually about 10% faster for small matrices to use MMatrix here instead of SMatrix.
        σ₁ = Σ[1]
        ϵᵢ = last(ds[i])

        # Scale semi-axis lengths to k-th neighbor distance
        rs .= ϵᵢ .* (Σ ./ σ₁)
        # Create matrix ellipse representation, centered at origin (fill Λ)
        hyperellipsoid_matrix!(Λ, Vt, rs)
        # In the paper the point itself is always counted inside the ellipsoid,
        # so that there is always one point present. Here we instead set the local density
        # to zero (by just skipping the computation) if that is the case.
        kᵢ = center_neighbors_and_count(neighborsᵢ, xᵢ, Λ)
        if kᵢ > 0
            h += log(kᵢ * γ / (f * ϵᵢ^D * prod(Σ ./ σ₁)) )
        end
    end
    # The estimated entropy has "unit" [nats]
    h = - h / N

    return convert_logunit(h, ℯ, est.definition.base)
end

# This is zero-allocating.
function fill_A!(A, C)
    @inbounds for (j, m) in enumerate(C)
        A[j, :] = m
    end
end

# Take the point `xᵢ` as the origin for the neighborhood, so we can check directly
# whether points are inside the ellipse. This happens for a point `x`
# whenever `xᵀΛx <= 1`.
function center_neighbors_and_count(neighborsᵢ, xᵢ, Λ)
    nns_centered = (pt - xᵢ for pt in neighborsᵢ)
    kᵢ = count(transpose(p) * Λ * p <= 1.0 for p in nns_centered)
    return kᵢ
end

# If all input vectors are `SVector`s, then this is zero-allocating.
function centroid(xᵢ::SVector{D}, neighbors, k::Int) where D
    centroid = SVector{D}(xᵢ)
    for nᵢ in neighbors
        centroid += nᵢ
    end
    centroid /= k + 1
    return centroid
end

# If all input vectors are `SVector`s, then this is zero-allocating.
"""
    center_neighborhood!(C, c, xᵢ, neighbors)

Center the point `xᵢ`, as well as each of its neighboring points `nⱼ ∈ neighbors`,
to the (precomputed) centroid `c` of the points `{xᵢ, n₁, n₂, …, nₖ}`, and store the
centered vectors in the pre-allocated vector of vectors `C`.
"""
function center_neighborhood!(C, c, xᵢ::SVector{D}, neighbors) where {D}
    C[1] = xᵢ - c
    for (i, nᵢ) in enumerate(neighbors)
        C[1 + i] = nᵢ - c
    end
    return C
end

function hyperellipsoid_matrix!(Λ, directions, extents)
    Λ .= 0.0
    for i in axes(directions, 2)
        vᵢ = directions[:, i]
        Λ .+= (vᵢ * transpose(vᵢ)) ./ extents[i]^2
    end
    return Λ
end
