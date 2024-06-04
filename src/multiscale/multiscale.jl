# This file contains an API for multiscale (coarse-grained/downsampled) computations.
# This API is not final nor agreed upon yet. Nothing is exported or documented
# as part of public API. It must be considered as work in progress.
export downsample
export multiscale, multiscale_normalized
export MultiScaleAlgorithm

using Statistics

"""
    MultiScaleAlgorithm

The supertype for all multiscale coarse-graining/downsampling algorithms. Concrete subtypes
are:

- [`RegularDownsampling`](@ref)
- [`CompositeDownsampling`](@ref)
"""
abstract type MultiScaleAlgorithm end

"""
    downsample(algorithm::MultiScaleAlgorithm, s::Int, x)

Downsample and coarse-grain `x` to scale `s` according to the given
[`MultiScaleAlgorithm`](@ref). The return type depends on `algorithm`.
"""
downsample(method::MultiScaleAlgorithm, s::Int, x)

downsample(alg::MultiScaleAlgorithm,  s::Int, x::AbstractStateSpaceSet) =
    StateSpaceSet(map(t -> downsample(alg, s, t)), columns(x)...)

"""
    multiscale(algorithm::MultiScaleAlgorithm, [args...], x; maxscale::Int = 8)

A convenience function to compute the multiscale version of any
[`InformationMeasureEstimator`](@ref) or [`ComplexityEstimator`](@ref).

## Description

This function generalizes tne multiscale entropy of [Costa2002](@cite) to any discrete
information measure, any differential information measure, and any other complexity measure.

It utilizes [`downsample`](@ref) with the given `algorithm` to first produce coarse-grained,
downsampled versions of `x` for scale factors `1:maxscale`. Then, [`information`](@ref) or
[`complexity`](@ref), depending on the input arguments, is applied to each of
the coarse-grained timeseries. If `N = length(x)`, then the length of the most severely
downsampled version of `x` is `N ÷ maxscale`, while for scale factor `1`, the original
time series is considered.

## Coarse-graining algorithms

The available downsampling routines are:

- [`RegularDownsampling`](@ref) yields a single `Vector` per scale.
- [`CompositeDownsampling`](@ref) yields a `Vector{Vector}` per scale.

## Keyword Arguments

- `maxscale::Int`. The maximum number of scales (i.e. levels of downsampling). The actual
    maximum scale level is `length(x) ÷ 2`, but to avoid applying the method to time
    series that are extremely short, maybe consider limiting `maxscale` (e.g.
    `maxscale = length(x) ÷ 5`).
"""
function multiscale end

"""
    multiscale_normalized(algorithm::MultiScaleAlgorithm, [args...], x; maxscale::Int = 8)

The same as [`multiscale`](@ref), but computes the normalized version of the complexity
measure.
"""
function multiscale_normalized end

max_scale_level(method::MultiScaleAlgorithm, x) = length(x) ÷ 2
function verify_scale_level(method, s::Int, x)
    err = DomainError(
        "Maximum scale for length-$(length(x)) timeseries is "*
        "`s = $(max_scale_level(method, x))`. Got s = $s"
    )
    length(x) ÷ s >= 2 || throw(err)
end

# To extend the multiscale interface to a new `MultiscaleAlgorithm`, simply extend this
# function for your new type.
"""
    apply_multiscale(alg::MultiScaleAlgorithm, f::Function, args...; maxscale = 8)

Define multiscale dispatch for the function `f` (either `information`, `complexity` or
their normalized variants) to downsampled timeseries resulting from coarse-graining
`last(args)` (the input data) using coarse-graining algorithm `alg` with arguments
`args[1:end-1]` (the estimation parameters).
"""
function apply_multiscale end

# Generate code for all possible `MultiscaleAlgorithms`s with all possible complexity
# measure quantifiers.
for fun = (:information, :complexity, :information_normalized, :complexity_normalized)
    @eval function $fun(multiscale_alg::MultiScaleAlgorithm, args...; maxscale = 8)
        define_multiscale(multiscale_alg, $fun, args...; maxscale)
    end
end

# Completely generic. Concrete implementations are in individual coarse-graining algorithm
# files, listed at the bottom of this file.
function multiscale(alg::MultiScaleAlgorithm, args...; maxscale = 8)
   f = infer_complexity_func(first(args); normalize = false) # measure is first argument
    return apply_multiscale(alg, f, args...; maxscale)
end
function multiscale_normalized(alg::MultiScaleAlgorithm, args...; maxscale = 8)
    f = infer_complexity_func(first(args); normalize = true)  # measure is first argument
    return apply_multiscale(alg, f, args...; maxscale)
end
function infer_complexity_func(T; normalize = false)
    if normalize
        if T isa InformationMeasureEstimator || T isa InformationMeasure
            return information_normalized
        elseif T isa ComplexityEstimator
            return complexity_normalized
        end
    else
        if T isa InformationMeasureEstimator || T isa InformationMeasure
            return information
        elseif T isa ComplexityEstimator
            return complexity
        end
    end
    throw("Measure type $T is not valid for normalize = $normalize")
end

include("regular.jl")
include("composite.jl")
