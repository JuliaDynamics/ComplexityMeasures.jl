export SequentialPairDistances

"""
    SequentialPairDistances <: CountBasedOutcomeSpace
    SequentialPairDistances(x; n = 3, m = 3, τ = 1, metric = Distances.Chebyshev())

An outcome space based on the distribution of distances of sequential pairs of points.

This outcome space appears implicitly as part of the "distribution entropy" introduced
by [Li2015](@citet), which of course can be reproduced here (see example below).
We've generalized the method to be used with any [`InformationMeasure`](@ref) and
[`DiscreteInfoEstimator`](@ref), and with valid distance `metric` (from Distances.jl).

Input data `x` are needed for initialization, because distances must be pre-computed to
know the minimum/maximum distances needed for binning the distribution of pairwise
distances. 

## Description

`SequentialPairDistances` does the following: 

- Transforms the input timeseries `x` by first embedding it using embedding dimension
    `m` and embedding lag `τ`.
- Computes the distances `ds` between sequential pairs of points according to the given
    `metric`.
- Divides the interval `[minimum(ds), maximum(ds)]` into `n` equal-size bins by using 
    [`RectangularBinEncoding`](@ref), then maps the distances onto these bins.

## Outcome space

The outcome space `Ω` for `SequentialPairDistances` are the bins onto which the 
pairwise distances are mapped, encoded as the integers `1:n`. If you need the actual
bin coordinates, these can be recovered with [`decode`](@ref) (see example below).

## Implements

- [`codify`](@ref). Note that the input `x` is ignored when calling `codify`, because
    the input data is already handled when constructing a `SequentialPairDistances`.

## Examples

The outcome bins can be retrieved as follows.

```julia
using ComplexityMeasures
x = rand(100)
o = SequentialPairDistances(x)
cts, outs = counts_and_outcomes(o, x)
```

Computing the "distribution entropy" with `n = 3` bins for the distance histogram:

```julia
using ComplexityMeasures
x = rand(1000000)
o = SequentialPairDistances(x, n = 3, metric = Chebyshev()) # metric from original paper
h = information(Shannon(base = 2), o, x)
```
"""
struct SequentialPairDistances{I<:Integer, M, T, DM, D, E} <: CountBasedOutcomeSpace
    n::I
    m::M
    τ::T
    metric::DM
    dists::D
    encoding::E
end

function SequentialPairDistances(x; n::I = 3, m::M = 3, τ::T = 1, 
    metric::DM = Chebyshev()) where {I, M, T, DM}
    x_embed = embed(x, m, τ)
    dists = [metric(x_embed[i], x_embed[i+1]) for i in 1:length(x)-1]
    mindist, maxdist = minimum(dists), maximum(dists)
    encoding = PairDistanceEncoding(mindist, maxdist; n, metric)
    D, E = typeof(dists), typeof(encoding)
    return new{I, M, T, DM, D, E}(n, m, τ, metric, dists, encoding)
end

# ----------------------------------------------------------------
# Pretty printing (see /core/pretty_printing.jl).
# ----------------------------------------------------------------
hidefields(::Type{<:SequentialPairDistances}) = [:dists]

total_outcomes(est::SequentialPairDistances) = est.n
outcome_space(est::SequentialPairDistances) = collect(1:est.n)

function counts_and_outcomes(o::SequentialPairDistances, x)
    return counts_and_outcomes(UniqueElements(), codify(o, x))
end

# ----------------------------------------------------------------
# A convenience iterator over pairwise points.
# ----------------------------------------------------------------
struct SequentialPairIterator{T}
    it::Vector{T}
end
function Base.iterate(p::SequentialPairIterator, i::Int=1)
    l = length(p.it)
    if i < l
        return (p.it[i], p.it[i + 1]), i + 1
    else
        return nothing
    end
end
Base.eltype(::Type{SequentialPairIterator{T}}) where {T} = Tuple{T,T}
Base.length(p::SequentialPairIterator) = length(p.it) - 1

function codify(o::SequentialPairDistances, x)
    return encode.(Ref(o.encoding.binencoder), o.dists)
end