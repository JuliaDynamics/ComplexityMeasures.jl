# This file contains an API for multiscale (coarse-grained/downsampled) computations.

using Statistics
export multiscale
export multiscale_normalized
export downsample
export MultiScaleAlgorithm

"""
    MultiScaleAlgorithm

The supertype for all multiscale algorithms.
"""
abstract type MultiScaleAlgorithm end

"""
    downsample(algorithm::MultiScaleAlgorithm, x, s::Int)

Downsample and coarse-grain `x` to scale `s` according to the given multiscale `algorithm`.

Positional arguments `args` and keyword arguments `kwargs` are propagated to relevant
functions in `algorithm`, if applicable.

The return type depends on `algorithm`. For example:

- [`Regular`](@ref) yields a single `Vector` per scale.
- [`Composite`](@ref) yields a `Vector{Vector}` per scale.
"""
downsample(method::MultiScaleAlgorithm, x, s::Int)

downsample(alg::MultiScaleAlgorithm, x::AbstractDataset, args...; kwargs...) =
    Dataset(map(t -> downsample(alg, t, args...; kwargs...), columns(x))...)


function multiscale end
function multiscale_normalized end

"""
    multiscale(e::Entropy, alg, x, est; kwargs...)

Compute the multi-scale entropy `e` with probabilities estimator `est` for timeseries `x`.

## Description

Utilizes [`downsample`](@ref) to compute the entropy/complexity of coarse-grained,
downsampled versions of `x` for scale factors `1:maxscale`. If `N = length(x)`, then the
length of the most severely downsampled version of `x` is `N ÷ scalemax`, while for scale
factor `1`, the original time series is considered.

If relevant [`MultiScaleAlgorithm`](@ref)s and corresponding [`downsample`](@ref)
methods are defined,
- the first method generalizes all multi-scale entropy estimators where actual entropies
    (i.e. functionals of explicitly estimated probability distributions) are computed, and
- the second method generalizes all multi-scale complexity ("entropy-like") measures.

## Arguments

- `e::Entropy`. A valid [entropy type](@ref entropies), i.e. `Shannon()` or `Renyi()`.
- `alg::MultiScaleAlgorithm`. A valid [multiscale algorithm](@ref multiscale_algorithms),
    i.e. `Regular()` or `Composite()`, which determines how down-sampling/coarse-graining
    is performed.
- `x`. The input data. Usually a timeseries.
- `est::ProbabilitiesEstimator`. A [probabilities estimator](@ref probabilities_estimators),
    which determines how probabilities are estimated for each downsampled time series.

## Keyword Arguments

- `scalemax::Int`. The maximum number of scales (i.e. levels of downsampling). The actual
    maximum scale level is `length(x) ÷ 2`, but the default is `length(x) ÷ 5`, to avoid
    computing the entropies for time series that are extremely short.
- `normalize::Bool`. If `normalize == true`, then (if possible) compute normalized
    entropy/complexity. If `normalize == false`, then compute the non-normalized measure.

[^Costa2002]: Costa, M., Goldberger, A. L., & Peng, C. K. (2002). Multiscale entropy
    analysis of complex physiologic time series. Physical review letters, 89(6), 068102.
"""
function multiscale(e::Entropy, alg::MultiScaleAlgorithm, x, est::ProbabilitiesEstimator)
    msg = "`multiscale` entropy not implemented for $e $est on data type $(typeof(x))"
    throw(ArgumentError(msg))
end

"""
    multiscale_normalized(e::Entropy, alg, x, est; maxscale = 8)

The same as [`multiscale`](@ref), but normalizes the entropy if [`entropy_maximum`](@ref)
is implemented for `e`.
"""
function multiscale_normalized(e::Entropy, alg::MultiScaleAlgorithm, x,
        est::ProbabilitiesEstimator)
    msg = "`multiscale_normalized` not implemented for $e $est on data type $(typeof(x))"
    throw(ArgumentError(msg))
end

max_scale_level(method::MultiScaleAlgorithm, x) = length(x) ÷ 2
function verify_scale_level(method, x, s::Int)
    err = DomainError(
        "Maximum scale for length-$(length(x)) timeseries is "*
        "`s = $(max_scale_level(method, x))`. Got s = $s"
    )
    length(x) ÷ s >= 2 || throw(err)
end


include("multiscale/regular.jl")
include("multiscale/composite.jl")
