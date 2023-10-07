export RectangularBinning, FixedRectangularBinning, AbstractBinning
export RectangularBinEncoding

"""
    AbstractBinning

Supertype encompassing [`RectangularBinning`](@ref) and [`FixedRectangularBinning`](@ref).
"""
abstract type AbstractBinning end
abstract type HistogramEncoding <: Encoding end

##################################################################
# Structs and docstrings
##################################################################
# The binning types are intermediate structs whose only purpose is
# instructions for initializing the `RectangularBinEncoding`

"""
    RectangularBinning(ϵ, precise = false) <: AbstractBinning

Rectangular box partition of state space using the scheme `ϵ`,
deducing the histogram extent and bin width from the input data.

`RectangularBinning` is a convenience struct.
It is re-cast into [`FixedRectangularBinning`](@ref)
once the data are provided, so see that docstring for info on the bin calculation
and the meaning of `precise`.

Binning instructions are deduced from the type of `ϵ` as follows:

1. `ϵ::Int` divides each coordinate axis into `ϵ` equal-length intervals
    that cover all data.
2. `ϵ::Float64` divides each coordinate axis into intervals of fixed size `ϵ`, starting
    from the axis minima until the data is completely covered by boxes.
3. `ϵ::Vector{Int}` divides the i-th coordinate axis into `ϵ[i]` equal-length
    intervals that cover all data.
4. `ϵ::Vector{Float64}` divides the i-th coordinate axis into intervals of fixed size
    `ϵ[i]`, starting from the axis minima until the data is completely covered by boxes.

`RectangularBinning` ensures all input data are covered by extending the created
ranges if need be.
"""
struct RectangularBinning{E} <: AbstractBinning
    ϵ::E
    precise::Bool
end
RectangularBinning(ε) = RectangularBinning(ε, false)

"""
    FixedRectangularBinning <: AbstractBinning
    FixedRectangularBinning(ranges::Tuple{<:AbstractRange...}, precise = false)

Rectangular box partition of state space where the partition along each dimension
is explicitly given by each range `ranges`, which is a tuple of `AbstractRange` subtypes.
Typically, each range is the output of the `range` Base function, e.g.,
`ranges = (0:0.1:1, range(0, 1; length = 101), range(2.1, 3.2; step = 0.33))`.
All ranges must be sorted.

The optional second argument `precise` dictates whether Julia Base's `TwicePrecision`
is used for when searching where a point falls into the range.
Useful for edge cases of points being almost exactly on the bin edges,
but it is exactly four times as slow, so by default it is `false`.

Points falling outside the partition do not contribute to probabilities.
Bins are always left-closed-right-open: `[a, b)`.
**This means that the last value of
each of the ranges dictates the last right-closing value.**
This value does _not_ belong to the histogram!
E.g., if given a range `r = range(0, 1; length = 11)`, with `r[end] = 1`,
the value `1` is outside the partition and would not attribute any
increase of the probability corresponding to the last bin (here `[0.9, 1)`)!

**Equivalently, the size of the histogram is
`histsize = map(r -> length(r)-1, ranges)`!**

`FixedRectangularBinning` leads to a well-defined outcome space without knowledge of input
data, see [`ValueBinning`](@ref).
"""
struct FixedRectangularBinning{R<:Tuple} <: AbstractBinning
    ranges::R
    precise::Bool
end

function FixedRectangularBinning(ranges::R, precise = false) where {R}
    if any(r -> !issorted(r), ranges)
        throw(ArgumentError("All input ranges must be sorted!"))
    end
    return FixedRectangularBinning{R}(ranges, precise)
end

"""
    FixedRectangularBinning(range::AbstractRange, D::Int = 1, precise = false)

This is a convenience method where each dimension of the binning has the same range
and the input data are `D` dimensional, which defaults to 1 (timeseries).
"""
function FixedRectangularBinning(r::AbstractRange, D::Int = 1, precise = false)
    FixedRectangularBinning(ntuple(x->r, D), precise)
end

"""
    RectangularBinEncoding <: Encoding
    RectangularBinEncoding(binning::RectangularBinning, x)
    RectangularBinEncoding(binning::FixedRectangularBinning)

An encoding scheme that [`encode`](@ref)s points `χ ∈ x` into their histogram bins.

The first call signature simply initializes a [`FixedRectangularBinning`](@ref)
and then calls the second call signature.

See [`FixedRectangularBinning`](@ref) for info on mapping points to bins.
"""
struct RectangularBinEncoding{R<:Tuple, D, T, C, L} <: HistogramEncoding
    ranges::R
    precise::Bool
    mini::SVector{D, T}
    widths::SVector{D, T}
    histsize::NTuple{D, Int}
    ci::C # cartesian indices
    li::L # linear indices
end

function Base.show(io::IO, x::RectangularBinEncoding)
    return print(io, "RectangularBinEncoding\n" *
        "  ranges: $(x.ranges)\n" *
        "  histogram size: $(x.histsize)"
    )
end

##################################################################
# Initialization of encoding
##################################################################
# Fixed grid
function RectangularBinEncoding(b::FixedRectangularBinning)
    ranges = b.ranges
    histsize = map(r -> length(r)-1, ranges)
    D = length(ranges)
    T = float(eltype(first(ranges)))
    mini = SVector{D,T}(map(minimum, ranges))
    widths = SVector{D,T}(map(step, ranges))
    ci = CartesianIndices(Tuple(histsize))
    li = LinearIndices(ci)
    RectangularBinEncoding(ranges, b.precise, mini, widths, histsize, ci, li)
end
function RectangularBinEncoding(b::FixedRectangularBinning, x)
    if length(b.ranges) != dimension(x)
        throw(ArgumentError("""
        The dimensionality of the `FixedRectangularBinning` and input `x` do not match.
        Got $(b.ranges) and $(dimension(x))."""))
    end
    return RectangularBinEncoding(b)
end

# Data-controlled grid: just cast into FixesRectangularBinning
function RectangularBinEncoding(b::RectangularBinning, x)
    RectangularBinEncoding(FixedRectangularBinning(b, x))
end
function FixedRectangularBinning(b::RectangularBinning, x)
    D = dimension(x)
    T = eltype(x)
    ϵ = b.ϵ
    mini, maxi = minmaxima(x)
    if ϵ isa AbstractFloat || ϵ isa AbstractVector{<:AbstractFloat}
        widths = SVector{D,T}(ϵ .* ones(SVector{D,T}))
        # To ensure all points are guaranteed to be covered, we add the width
        # to the max, if the max isn't included in the resulting range.
        # We also add the width if the maximum is the end point of the range,
        # as according to our definition, the end point of a range is NOT
        # included in the histogram!
        ensure_covering_range = (i) -> begin
            r = range(mini[i], maxi[i]; step = widths[i])
            if (maxi[i] ∉ r) || (maxi[i] == r[end])
                return range(mini[i], maxi[i] + widths[i]; step = widths[i])
            else
                return r
            end
        end
        ranges = ntuple(i -> ensure_covering_range(i), D)
    elseif ϵ isa Int || ϵ isa Vector{Int}
        # use `nextfloat` to ensure all data are covered
        if b.precise # in the precise case we can just use 1 next float
            maxi = nextfloat.(maxi)
        else # otherwise we need a bit more width
            maxi = nextfloat.(maxi, 2)
        end
        # We add one, because the user input specifies the number of bins,
        # and the number of bins is the range length - 1
        lengths = ϵ .* ones(SVector{D,Int}) .+ 1
        ranges = ntuple(i -> range(mini[i], maxi[i]; length = lengths[i]), D)
    else
        error("Invalid ϵ for binning of a dataset")
    end
    # By default we have the imprecise version here;
    # use `Fixed` if you want precise
    return FixedRectangularBinning(ranges, b.precise)
end

##################################################################
# encode/decode
##################################################################
function encode(e::RectangularBinEncoding, point)
    cartidx = cartesian_bin_index(e, point)
    # We have decided on the arbitrary convention that out of bound points
    # will get the special symbol `-1`. Erroring doesn't make sense as it is expected
    # that for fixed histograms there may be points outside of them.
    if checkbounds(Bool, e.li, cartidx)
        return @inbounds e.li[cartidx]
    else
        return -1
    end
end

"""
    cartesian_bin_index(e::RectangularBinEncoding, point::SVector)

Return the cartesian index of
the given `point` within the binning encapsulated in `e`.
Internal function called by `encode`.
"""
function cartesian_bin_index(e::RectangularBinEncoding, point)
    ranges = e.ranges
    if e.precise
        # Don't know how to make this faster unfurtunately...
        cartidx = CartesianIndex(map(searchsortedlast, ranges, Tuple(point)))
    else
        bin = floor.(Int, (point .- e.mini) ./ e.widths) .+ 1
        cartidx = CartesianIndex(Tuple(bin))
    end
    return cartidx
end

function decode(e::RectangularBinEncoding, bin::Integer)
    if checkbounds(Bool, e.ci, bin)
        @inbounds cartesian = e.ci[bin]
    else
        throw(ArgumentError(
            "Cannot decode integer $(bin): out of bounds of underlying binning."
        ))
    end
    # The decoding step is rather trivial here; we just index the ranges at the index
    ranges = e.ranges
    D = length(ranges)
    left_edge = @inbounds ntuple(i -> ranges[i][cartesian[i]], D)
    return SVector{D}(left_edge)
    # widths = e.widths
    # mini = e.mini
    # V = SVector{length(ranges), eltype(float(first(ranges)))}
    # Remove one because we want lowest value corner, and we get indices starting from 1
    # return (V(Tuple(cartesian)) .- 1) .* widths .+ mini
end

##################################################################
# Outcomes / total outcomes
##################################################################
total_outcomes(e::RectangularBinEncoding) = prod(e.histsize)

function outcome_space(e::RectangularBinEncoding)
    # This is super simple thanks to using ranges :)
    reduced_ranges = map(r -> r[1:end-1], e.ranges)
    iter = Iterators.product(reduced_ranges...)
    # Convert to `SVector` because that's the agreed outcome space type
    V = SVector{length(e.ranges), eltype(float(first(e.ranges)))}
    return sort!(vec(V.(iter)))
end
outcome_space(b::AbstractBinning, args...) =
outcome_space(RectangularBinEncoding(b, args...))

##################################################################
# low level histogram call
##################################################################
# This method is called by `probabilities(est::ValueBinning, x::Array_or_SSSet)`
# `fasthist!` is in the `estimators/value_histogram` folder.
"""
    fasthist(c::RectangularBinEncoding, x::Vector_or_SSSet)
Intermediate method that runs `fasthist!` in the encoded space
and returns the encoded space histogram (counts) and corresponding bins.
Also skips any instances of out-of-bound points for the histogram.
"""
function fasthist(encoder::RectangularBinEncoding, x)
    bins = map(y -> encode(encoder, y), x)
    # We discard `-1`, as it encodes points outside the histogram limit
    # (which should only happen for `Fixed` binnings)
    discard_minus_ones!(bins)
    hist = fasthist!(bins)
    return hist, bins
end

function discard_minus_ones!(bins)
    idxs = findall(isequal(-1), bins)
    deleteat!(bins, idxs)
end

# ----------------------------------------------------------------
# Convenience compatibility with counting api.
# ----------------------------------------------------------------
function counts(encoding::RectangularBinEncoding, x)
    cts, bins = fasthist(encoding, x) # bins are integers here
    unique!(bins) # `bins` is already sorted from `frequencies!`
    # Here we transform the cartesian coordinate based bins into data unit bins:
    outcomes = map(b -> decode(encoding, b), bins)
    return Counts(cts, (outcomes,))
end
