#export Pal
using Neighborhood
using StateSpaceSets: dimension, AbstractStateSpaceSet, StateSpaceSet
export Pal

"""
    Pal <: <: DifferentialInfoEstimator
    Pal(; k = 1, w = 0, p = 1.0, n::Int = 10000)

A [`Shannon`](@ref] and [`Renyi`](@ref) differential entropy estimator [Pal2010](@cite)
based on `k`-th nearest neighbor searches.

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

## Description

[Pal2010](@citet)'s estimator is based on generalized nearest neighbor graphs. It is
similar to several other kNN-based estimators (e.g. [`LeonenkoProzantoSavani`](@ref)).
Given samples ``\\bf{X}_{1:n} = \\{\\bf{X}_1, \\bf{X}_2, \\ldots, \\bf{X}_n \\}``
where ``\\bf{X}_1 \\in \\mathbb{R}^d`` from some distribution ``\\mu`` over
``\\mathbb{R}^d`` with density function ``f``,
approximates the [`Renyi`](@ref) differential entropy

```math
h_q^R(\\bf(X)) = \\dfrac{1}{1-q} \\int_{\\mathbb{R}^d} f^q(\\bf{x}) d\\bf{x}
```

using the estimator

```math
\\hat{H}_q^R(\\bf{X_{1:n}}) = \\dfrac{1}{1-q}\\log
\\left( \\dfrac{L_p(\\bf{X}_{1:n})}{\\gamma n^{1-p/d}} \\right),
```

where ``L_p(\\bf{X}_{1:n}`` is the sum of the `p`-th powers of the Euclidean
lengths of the edges of the nearest neighbor graph over ``\\bf{X}_{1:n}``
(see their paper for details).

The constant ``\\gamma`` is determined by the limit given in equation 4 in
Pàl et al. (2010),
and is approximated on `n` randomly generated points from the `d`-dimensional
unit cube, as they describe in the end of section 3 of their paper.
"""
struct Pal{I, P} <: DifferentialInfoEstimator{I}
    definition::I
    k::Int
    w::Int
    p::P
    n::Int
    function Pal(definition::I, k, w, p::P, n) where {I, P}
        if !(I <: Shannon || (I <: Renyi))
            s = "`definition` must be either a `Shannon` or `Renyi` instance."
            throw(ArgumentError(s))
        end
        if k <= 1
            throw(ArgumentError("`k` must be ≥ 2. Got k=$k."))
        end

        new{I, P}(definition, k, w, p, n)
    end
end

function Pal(definition = Shannon(); k = 2, w = 0, p = 2.0, n = 5000)
    return Pal(definition, k, w, p, n)
end

function information(est::Pal, x::AbstractVector{<:Real})
    return information(est, StateSpaceSet(x))
end

function information(est::Pal{<:Renyi}, x::AbstractStateSpaceSet)
    (; definition, k, w, p, n) = est
    (; q, base) = definition
    q <= 1 || error("Pal estimator only defined for 0 < q <= 1")
    if q == 1
        q = 0.999999999
    end

    d = dimension(x)
    γ = approximate_γ(est, d)
    h = 1 / (1 - q) * log(Lₚ(est, x) / (γ * n^(1 - p/d)))
    return convert_logunit(h, ℯ, base)
end

function information(est::Pal{<:Shannon}, x::AbstractStateSpaceSet)
    (; definition, k, w, p, n) = est
    (; base) = definition
    q = 1.0 - eps() # approximate Shannon entropy by simply letting q → 1
    X = StateSpaceSet(x)
    N = length(x)
    d = dimension(X)
    γ = approximate_γ(est, d)
    L = Lₚ(est, X)
    f = L / (γ * N^(1 - p/d))
    h = (1 / (1 - q)) * log(f)
    return convert_logunit(h, ℯ, base)
end

function Lₚ(est::Pal, x::AbstractStateSpaceSet{D}) where D
    (; k, w, p, n) = est
    N = length(x)
    tree = KDTree(x, Euclidean())
    idxs, ds = bulksearch(tree, x, NeighborNumber(k), Theiler(w))
    Lₚ = 0.0
    for i = 1:N
        Lₚ += sum(ds[i] .^ p)
    end
    return Lₚ
end

# TODO: Estimates of `γ` typically don't stabilize until millions of points are
# included. Thus, for practical applications where parameters of the
# analysis are allowed to vary, runtime quickly increases.
# Fitting a function f(d, p, k) would dramatically reduce runtime (how?).
# Alternatively, providing a look-up table for "commonly used" (as we define how we
# see fit) parameters ranges.
function approximate_γ(est::Pal, d::Int)
    p = est.p
    n = est.n
    x = StateSpaceSet(rand(n, d))
    return Lₚ(est, x) / n^(1 - (p/d))
end
