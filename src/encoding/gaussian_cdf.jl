using Statistics, QuadGK

export GaussianCDFEncoding

"""
    GaussianCDFEncoding <: Encoding
    GaussianCDFEncoding(; μ, σ, c::Int = 3)

An encoding scheme that [`encode`](@ref)s a scalar value into one of the integers
`sᵢ ∈ [1, 2, …, c]` based on the normal cumulative distribution function (NCDF),
and [`decode`](@ref)s the `sᵢ` into subintervals of `[0, 1]` (with some loss of information).

## Algorithm

`GaussianCDFEncoding` first maps an input point ``x``  (scalar) to a new real number
``y_ \\in [0, 1]`` by using the normal cumulative distribution function (CDF) with the
given mean `μ` and standard deviation `σ`, according to the map

```math
x \\to y : y = \\dfrac{1}{ \\sigma
    \\sqrt{2 \\pi}} \\int_{-\\infty}^{x} e^{(-(x - \\mu)^2)/(2 \\sigma^2)} dx.
```

Next, the interval `[0, 1]` is equidistantly binned and enumerated ``1, 2, \\ldots, c``.
 and ``y`` is linearly mapped to one of these integers using the linear map
 ``y \\to z : z = \\text{floor}(y(c-1)) + 1``.

Because of the ceiling operation, some information is lost, so when used with
[`decode`](@ref), each decoded `sᵢ` is mapped to a *subinterval* of `[0, 1]`.

## Usage

In the implementations of [`Dispersion`](@ref) and [`RelativeDispersion`](@ref),
we use the empirical means `μ` and standard deviation `σ`, as determined from
a univariate input timeseries ``X = \\{x_i\\}_{i=1}^N``, and after encoding, the input
is thus transformed to a symbol time
series ``S = \\{ s_i \\}_{i=1}^N``, where ``s_i \\in [1, 2, \\ldots, c]``.

## Examples

```jldoctest
julia> using Entropies, Statistics

julia> x = [0.1, 0.4, 0.7, -2.1, 8.0, 0.9, -5.2];

julia> μ, σ = mean(x), std(x); encoding = GaussianCDFEncoding(; μ, σ, c = 5)

julia> es = Entropies.encode.(Ref(encoding), x)
7-element Vector{Int64}:
 3
 3
 3
 2
 5
 3
 1

julia ds = Entropies.decode.(Ref(encoding), es)
```
"""
struct GaussianCDFEncoding{T} <: Encoding
    c::Int
    σ::T
    μ::T
    # We require the input data, because we need σ and μ for encoding single values.
    function GaussianCDFEncoding(; μ::T, σ::T, c::Int = 3) where T
        new{T}(c, σ, μ)
    end
end

total_outcomes(encoding::GaussianCDFEncoding) = encoding.c

gaussian(x, μ, σ) = exp((-(x - μ)^2)/(2σ^2))

function encode(encoding::GaussianCDFEncoding, x::Real)
    (; c, σ, μ) = encoding
    # We only need the value of the integral (not the error), so
    # index first element returned from quadgk
    k = 1/(σ*sqrt(2π))
    y = k * first(quadgk(x -> gaussian(x, μ, σ), -Inf, x))
    return floor(Int, y / (1 / c)) + 1
end

function decode(encoding::GaussianCDFEncoding, i::Int)
    c = encoding.c
    V = SVector{2, eltype(1.0)}
    lower_interval_bound = (i - 1)/(c)

    yⱼ = V(lower_interval_bound, lower_interval_bound + 1/c - eps())
end
