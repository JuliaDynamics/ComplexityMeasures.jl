export AmplitudeEncoding

"""
    AmplitudeEncoding <: Encoding
    AmplitudeEncoding(minval::Real, maxval::Real; n = 2)

Encoding which discretizes a state vector (an `AbstractVector` of some kind) into its
"absolute amplitude symbol"  relative to some pre-dedined minimum value `minval` and
maximum value `maxval` (see description below).

## Description

This encoding is inspired by Azami & Escudero[^Azami2016]'s algorithm for amplitude-aware
permutation entropy. They use a linear combination of amplitude information and
first differences information of state vectors to correct probabilities. Here, however,
we explicitly encode the amplitude-part of the correction as an a integer symbol
`Λ ∈ [1, 2, …, n]`. The first-difference part of the encoding is available
as the [`FirstDifferenceEncoding`](@ref) encoding.

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
struct AmplitudeEncoding <: Encoding
    n::Int
    minval::Real
    maxval::Real
    encoder::RectangularBinEncoding

    function AmplitudeEncoding(minval::Real, maxval::Real; n::Int = 2)
        encoder = RectangularBinEncoding(FixedRectangularBinning(0, 1, n + 1))

        if minval > maxval
            s = "Need minval <= maxval. Got minval=$minval and maxval=$maxval."
            throw(ArgumentError(s))
        end
        return new(n, minval, maxval, encoder)
    end
end

function encode(encoding::AmplitudeEncoding, x::AbstractVector)
    Λ = sum(abs(xᵢ) for xᵢ in x) / length(x)
    Λ_normalized = norm_minmax(Λ, encoding.minval, encoding.maxval)

    # Return an integer from the set {1, 2, …, encoding.n}
    return encode(encoding.encoder, Λ_normalized)
end

function decode(encoding::AmplitudeEncoding, ω::Int)
    # Return the left-edge of the bin.
    return decode(encoding.encoder, ω)
end

total_outcomes(encoding::AmplitudeEncoding) = total_outcomes(encoding.encoder)
