export RelativeMeanEncoding

"""
    RelativeMeanEncoding <: Encoding
    RelativeMeanEncoding(minval::Real, maxval::Real; n = 2)

`RelativeMeanEncoding` encodes a vector based on the relative position the mean of the
vector has with respect to a predefined minimum and maximum value (`minval` and
`maxval`, respectively).

## Description

This encoding is inspired by Azami & Escudero[^Azami2016]'s algorithm for amplitude-aware
permutation entropy. They use a linear combination of amplitude information and
first differences information of state vectors to correct probabilities. Here, however,
we explicitly encode the amplitude-part of the correction as an a integer symbol
`Λ ∈ [1, 2, …, n]`. The first-difference part of the encoding is available
as the [`RelativeFirstDifferenceEncoding`](@ref) encoding.

## Encoding/decoding

When used with [`encode`](@ref), an ``m``-element state vector
``\\bf{x} = (x_1, x_2, \\ldots, x_m)`` is encoded
as ``Λ = \\dfrac{1}{N}\\sum_{i=1}^m abs(x_i)``. The value of ``Λ`` is then normalized
to lie on the interval `[0, 1]`, assuming that the minimum/maximum value any single
element ``x_i`` can take is `minval`/`maxval`, respectively. Finally, the interval
`[0, 1]` is discretized into `n` discrete bins, enumerated by positive integers
`1, 2, …, n`, and the number of the bin that the normalized ``Λ`` falls into is returned.

When used with [`decode`](@ref), the left-edge of the bin that the normalized ``Λ``
fell into is returned.

[^Azami2016]:
    Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy:
    Illustration in spike detection and signal segmentation. Computer methods and
    programs in biomedicine, 128, 40-51.
"""
struct RelativeMeanEncoding{R} <: Encoding
    n::Int
    minval::Real
    maxval::Real
    binencoder::R # RectangularBinEncoding

    function RelativeMeanEncoding(n::Int, minval::Real, maxval::Real, binencoder::R) where R
        if minval > maxval
            s = "Need minval <= maxval. Got minval=$minval and maxval=$maxval."
            throw(ArgumentError(s))
        end
        if n < 1
            throw(ArgumentError("n must be ≥ 1"))
        end
        return new{typeof(binencoder)}(n, minval, maxval, binencoder)
    end
end

function Base.show(io::IO, e::RelativeMeanEncoding)
    n, minval, maxval = e.n, e.minval, e.maxval
    print(io, "RelativeMeanEncoding(n=$n, minval=$minval, maxval=$maxval)")
end

function RelativeMeanEncoding(minval::Real, maxval::Real; n = 2)
    binencoder = RectangularBinEncoding(FixedRectangularBinning(0, 1, n + 1))
    return RelativeMeanEncoding(n, minval, maxval, binencoder)
end

function encode(encoding::RelativeMeanEncoding, x::AbstractVector)
    (; n, minval, maxval, binencoder) = encoding
    Λ = sum(abs(xᵢ) for xᵢ in x) / length(x)

    # Normalize to [0, 1]
    Λ_normalized = (Λ - minval) / (maxval - minval)

    # Return an integer from the set {1, 2, …, encoding.n}
    return encode(binencoder, Λ_normalized)
end

function decode(encoding::RelativeMeanEncoding, ω::Int)
    # Return the left-edge of the bin.
    return decode(encoding.binencoder, ω)
end

total_outcomes(encoding::RelativeMeanEncoding) = total_outcomes(encoding.binencoder)
