export NaiveKernel, genentropy, probabilities

using NearestNeighbors
using DelayEmbeddings

"""
    NaiveKernel(ϵ::Real) <: ProbabilitiesEstimator

Estimate probabilities/entropy using a kernel density estimation (KDE). This is the "naive kernel" approach 
discussed in Prichard and Theiler (1995) [^PrichardTheiler1995].

Probabilities ``P(\\mathbf{x}, \\epsilon)`` are assigned to every point ``\\mathbf{x}`` by counting how many other points occupy the space spanned by 
a hypersphere of radius `ϵ` around ``\\mathbf{x}``, according to:

```math
P_i(\\mathbf{x}, \\epsilon) \\approx \\dfrac{1}{N} 
\\sum_{s \\neq i } K\\left(\\dfrac{||\\mathbf{x}_i - \\mathbf{x}_s ||}{\\epsilon} \\right),
```

where ``K(z) = 1`` if ``z < 1`` and zero otherwise. Probabilities are sum-normalized after 
the ``P_i(\\mathbf{x}, \\epsilon)`` have been obtained.

[^PrichardTheiler1995]: Prichard, D., & Theiler, J. (1995). Generalized redundancies for time series analysis. Physica D: Nonlinear Phenomena, 84(3-4), 476-493.
"""
struct NaiveKernel <: ProbabilitiesEstimator
    ϵ::Real
    function NaiveKernel(ϵ::Real)
        ϵ > 0 || error("Radius ϵ must be larger than zero, otherwise no radius around the points is defined!")
        new(ϵ)
    end

    function NaiveKernel()
        error("NaiveKernel takes one input argument: the radius ϵ.")
    end
end

function get_pts_within_radius(pts, est::NaiveKernel)
    tree = KDTree(pts)
    
    # Count number of points within radius of ϵ for each point in `pts`
    # There's no need to sort the indices, we just count the number of points within radius ϵ.
    idxs = inrange(tree, pts.data, est.ϵ, false) 
    return [length(idx) for idx in idxs]
end

"""
    probabilities(x::AbstractDataset, est::NaiveKernel)

Estimate probabilities around each of the points in `x` using the `NaiveKernel` estimator.
"""
function probabilities(x::AbstractDataset, est::NaiveKernel)
    N = length(x)
    idxs = get_pts_within_radius(x, est) ./ N
    return idxs ./ sum(idxs)
end

"""
    genentropy(x::AbstractDataset, est::NaiveKernel; α::Real = 1, base = 2)

Estimate order-`α` generalized entropy from the data `x` using the `NaiveKernel` estimator.
"""
function genentropy(x::AbstractDataset, est::NaiveKernel; α::Real = 1, base = 2)
    N = length(x)
    ps = probabilities(x, est)
    return genentropy(α, ps, base = base)
end