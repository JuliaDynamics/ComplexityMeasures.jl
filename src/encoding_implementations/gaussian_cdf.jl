using Statistics, QuadGK

export GaussianCDFEncoding

"""
    GaussianCDFEncoding <: Encoding
    GaussianCDFEncoding(m::Int = 1; μ, σ, c::Int = 3)
    GaussianCDFEncoding(χ::AbstractVector; μ, σ, c::Int = 3)

An encoding scheme that [`encode`](@ref)s a scalar or vector `χ` into one of the integers
`sᵢ ∈ [1, 2, …, c]` based on the normal cumulative distribution function (NCDF),
and [`decode`](@ref)s the `sᵢ` into subintervals of `[0, 1]` (with some loss of information).

The size of the input to be encoded must be known beforehand, and one must set
`m = length(χ)`, where `χ` is the input (`m = 1` for scalars, `m ≥ 2` for vectors).
Alternatively, provide the vector `χ` to the constructor to infer `m` automatically.

Notice that the decoding step does not yield an element of any outcome space of the
estimators that use `GaussianCDFEncoding` internally, such as [`Dispersion`](@ref).
That is because these estimators additionally delay embed the encoded data.

## Description

### Encoding/decoding scalars

`GaussianCDFEncoding` first maps an input scalar ``χ`` to a new real number
``y_ \\in [0, 1]`` by using the normal cumulative distribution function (CDF) with the
given mean `μ` and standard deviation `σ`, according to the map

```math
x \\to y : y = \\dfrac{1}{ \\sigma
    \\sqrt{2 \\pi}} \\int_{-\\infty}^{x} e^{(-(x - \\mu)^2)/(2 \\sigma^2)} dx.
```

Next, the interval `[0, 1]` is equidistantly binned and enumerated ``1, 2, \\ldots, c``,
 and ``y`` is linearly mapped to one of these integers using the linear map
 ``y \\to z : z = \\text{floor}(y(c-1)) + 1``.

Because of the floor operation, some information is lost, so when used with
[`decode`](@ref), each decoded `sᵢ` is mapped to a *subinterval* of `[0, 1]`.
This subinterval is returned as a length-`1` `Vector{SVector}`.

### Encoding/decoding vectors

If `GaussianCDFEncoding` is used with a vector `χ`, then each element of `χ` is
encoded separately, resulting in a `length(χ)` sequence of integers which may be
treated as a `CartesianIndex`. The encoded symbol `s ∈ [1, 2, …, c]` is then just the
linear index corresponding to this cartesian index (similar to how
[CombinationEncoding](@ref) works).

When [`decode`](@ref)d, the integer symbol `s` is converted back into its `CartesianIndex`
representation,  which is just a sequence of integers that refer to subdivisions
of the `[0, 1]` interval. The relevant subintervals are then returned as a length-`χ`
`Vector{SVector}`.

## Examples

```jldoctest
julia> using ComplexityMeasures, Statistics

julia> x = [0.1, 0.4, 0.7, -2.1, 8.0];

julia> μ, σ = mean(x), std(x); encoding = GaussianCDFEncoding(; μ, σ, c = 5)

julia> es = encode.(Ref(encoding), x)
5-element Vector{Int64}:
 2
 2
 3
 1
 5

julia> decode(encoding, 3)
2-element SVector{2, Float64} with indices SOneTo(2):
 0.4
 0.6
```
"""
struct GaussianCDFEncoding{m, T, L <: LinearIndices, C <: CartesianIndices, R} <: Encoding
    m::Int
    c::Int
    σ::T
    μ::T

    # internal fields: LinearIndices/CartesianIndices for encodings/decodings. binencoder
    # for discretizing the interval [0, 1]
    linear_indices::L
    cartesian_indices::C
    binencoder::R # RectangularBinEncoding

    # The input `m` restricts what length the input scalar/vector can be.
    function GaussianCDFEncoding(m::Int = 1; μ::T, σ::T, c::Int = 3) where T
        m >= 1 || throw(ArgumentError("m must be an integer ≥ 1. Got $m."))
        ranges = tuple([1:c for i in 1:m]...)
        cartesian_indices = CartesianIndices(ranges)
        linear_indices = LinearIndices(ranges)
        L = typeof(linear_indices)
        C = typeof(cartesian_indices)
        binencoder = RectangularBinEncoding(FixedRectangularBinning(0, 1, c + 1))
        R = typeof(binencoder)
        new{m, T, L, C, R}(m, c, σ, μ, linear_indices, cartesian_indices, binencoder)
    end
end
GaussianCDFEncoding(x::AbstractVector; kwargs...) = GaussianCDFEncoding(length(x); kwargs...)

function Base.show(io::IO, e::GaussianCDFEncoding{m, T, L, C}) where {m, T, L, C}
    c, μ, σ = e.c, e.μ, e.σ
    print(io, "GaussianCDFEncoding(m=$m; c=$c, μ=$μ, σ=$σ)")
end

function total_outcomes(encoding::GaussianCDFEncoding{m}) where m
    c = encoding.c
    return c^m
end

gaussian(x, μ, σ) = exp((-(x - μ)^2)/(2σ^2))

function encode(encoding::GaussianCDFEncoding, x::Real)
    σ, μ = encoding.σ, encoding.μ
    # We only need the value of the integral (not the error), so
    # index first element returned from quadgk
    k = 1/(σ*sqrt(2π))
    y = k * first(quadgk(x -> gaussian(x, μ, σ), -Inf, x))
    # The integral estimate sometime returns a value slightly above 1.0, so we need
    # to adjust to be sure that all points fall within the FixedRectangularBinning.
    y_corrected = min(y, 1.0)
    return encode(encoding.binencoder, y_corrected)
end

function encode(encoding::GaussianCDFEncoding{m}, x::AbstractVector) where m
    L = length(x)
    if L != m
        throw(ArgumentError("length(`x`) must equal `m` (got length(x)=$L, m=$m)"))
    end
    symbols = encode.(Ref(encoding), x)
    ω::Int = encoding.linear_indices[symbols...]
    return ω
end

function decode(encoding::GaussianCDFEncoding, ω::Int)
    cidxs = Tuple(encoding.cartesian_indices[ω])
    return [decode(encoding.binencoder, cᵢ) for cᵢ in cidxs]
end
