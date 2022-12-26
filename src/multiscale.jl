# This file contains an API for multiscale (coarse-grained/downsampled) computations.
# This API is not final nor agreed upon yet. Nothing is exported or documented
# as part of public API. It must be considered as work in progress.

using Statistics

"""
    MultiScaleAlgorithm

The supertype for all multiscale algorithms.
"""
abstract type MultiScaleAlgorithm end

"""
    downsample(algorithm::MultiScaleAlgorithm, s::Int, x)

Downsample and coarse-grain `x` to scale `s` according to the given multiscale `algorithm`.

Positional arguments `args` and keyword arguments `kwargs` are propagated to relevant
functions in `algorithm`, if applicable.

The return type depends on `algorithm`. For example:

- [`Regular`](@ref) yields a single `Vector` per scale.
- [`Composite`](@ref) yields a `Vector{Vector}` per scale.
"""
downsample(method::MultiScaleAlgorithm, s::Int, x)

downsample(alg::MultiScaleAlgorithm,  s::Int, x::AbstractDataset) =
    Dataset(map(t -> downsample(alg, s, t)), columns(x)...)


function multiscale end
function multiscale_normalized end

"""
    multiscale(alg::MultiScaleAlgorithm, e::EntropyDefinition, est::DiffEntropyEst, x; kwargs...)
    multiscale(alg::MultiScaleAlgorithm, e::EntropyDefinition, est::ProbabilitiesEstimator, x; kwargs...)
    multiscale(alg::MultiScaleAlgorithm, c::ComplexityMeasure, x::AbstractVector; kwargs...)

Compute the multi-scale entropy `e` with estimator `est`, or the complexity measure `c`,
for timeseries `x`.

The first signature estimates differential/continuous multiscale entropy. The second
signature estimates discrete multiscale entropy. The third signature estimate some
other multiscale complexity measure that is not, strictly speaking, an entropy.

This function generalizes *all* multi-scale entropy estimators, as long as a relevant
[`MultiScaleAlgorithm`](@ref), a [`downsample`](@ref) method and an estimator is defined.
It also generalizes all multi-scale complexity measures, provided a relevant
[`ComplexityMeasre`](@ref) is defined.

## Description

Utilizes [`downsample`](@ref) to compute the entropy/complexity of coarse-grained,
downsampled versions of `x` for scale factors `1:maxscale`. If `N = length(x)`, then the
length of the most severely downsampled version of `x` is `N ÷ maxscale`, while for scale
factor `1`, the original time series is considered.

## Arguments

- `e::EntropyDefinition`. A valid [entropy type](@ref entropies), i.e. `Shannon()` or `Renyi()`.
- `c::ComplexityMeasure`. A valid complexity measure, i.e. [`SampleEntropy`](@ref),
    or [`ApproximateEntropy`](@ref).
- `alg::MultiScaleAlgorithm`. A valid [multiscale algorithm](@ref multiscale_algorithms),
    i.e. `Regular()` or `Composite()`, which determines how down-sampling/coarse-graining
    is performed.
- `x`. The input data. Usually a timeseries.
- `est`. For discrete entropy, `est` is a [`ProbabilitiesEstimator`](@ref), which determines
    how probabilities are estimated for each sampled time series. Alternatively,for
    continuous/differential entropy, `est` can be a [`DiffEntropyEst`](@ref),
    which dictates the entropy estimation method for each downsampled time series.

## Keyword Arguments

- `maxscale::Int`. The maximum number of scales (i.e. levels of downsampling). The actual
    maximum scale level is `length(x) ÷ 2`, but to avoid applying the method to time
    series that are extremely short, maybe consider limiting `maxscale` (e.g.
    `maxscale = length(x) ÷ 5`).

[^Costa2002]: Costa, M., Goldberger, A. L., & Peng, C. K. (2002). Multiscale entropy
    analysis of complex physiologic time series. Physical review letters, 89(6), 068102.
"""
function multiscale(alg::MultiScaleAlgorithm, e::EntropyDefinition,
        est::Union{ProbabilitiesEstimator, DiffEntropyEst}, x)
    msg = "`multiscale` entropy not implemented for $e $est on data type $(typeof(x))"
    throw(ArgumentError(msg))
end

"""
    multiscale_normalized(alg::MultiScaleAlgorithm, e::EntropyDefinition,
        est::ProbabilitiesEstimator, x)

The same as [`multiscale`](@ref), but normalizes the entropy if [`entropy_maximum`](@ref)
is implemented for `e`.

Note: this doesn't work if `est` is an [`DiffEntropyEst`](@ref).
"""
function multiscale_normalized(alg::MultiScaleAlgorithm, e::EntropyDefinition, est, x)
    msg = "`multiscale_normalized` not implemented for $e $(typeof(est)) on data type $(typeof(x))"
    throw(ArgumentError(msg))
end

multiscale_normalized(alg::MultiScaleAlgorithm, e::EntropyDefinition, est::DiffEntropyEst, x) =
    error("multiscale_normalized not defined for $(typeof(est))")

max_scale_level(method::MultiScaleAlgorithm, x) = length(x) ÷ 2
function verify_scale_level(method, s::Int, x)
    err = DomainError(
        "Maximum scale for length-$(length(x)) timeseries is "*
        "`s = $(max_scale_level(method, x))`. Got s = $s"
    )
    length(x) ÷ s >= 2 || throw(err)
end


include("multiscale/regular.jl")
include("multiscale/composite.jl")
