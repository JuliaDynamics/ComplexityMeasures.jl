export FirstDifferenceEncoding

"""
    FirstDifferenceEncoding <: Encoding
    FirstDifferenceEncoding(minval::Real, maxval::Real; n = 2)

Encoding which discretizes a state vector (an `AbstractVector` of some kind) into its
"first difference symbol"  relative to some pre-defined minimum (`minval`) and
maximum (`maxval`) first difference (see description below).

## Description

This encoding is inspired by Azami & Escudero[^Azami2016]'s algorithm for amplitude-aware
permutation entropy. They use a linear combination of amplitude information and
first differences information of state vectors to correct probabilities. Here, however,
we explicitly encode the first differences part of the correction as an a integer symbol
`Λ ∈ [1, 2, …, n]`. The amplitude part of the encoding is available
as the [`AmplitudeEncoding`](@ref) encoding.

## Encoding/decoding

When used with [`encode`](@ref), an ``m``-element state vector
``\\bf{x} = (x_1, x_2, \\ldots, x_m)`` is encoded
as ``Λ = \\dfrac{1}{m - 1}\\sum_{k=2}^m |x_{k} - x_{k-1}|``. The value of ``Λ`` is then
normalized to lie on the interval `[0, 1]`, assuming that the minimum/maximum value any
single ``abs(x_k - x_{k-1})`` can take is `minval`/`maxval`, respectively. Finally, the
interval `[0, 1]` is discretized into `n` discrete bins, enumerated by positive integers
`1, 2, …, n`, and the number of the bin that the normalized ``Λ`` falls into is returned.
The smaller the mean first difference of the state vector is, the smaller the bin number is.
The higher the mean first difference of the state vectors is, the higher the bin number is.

When used with [`decode`](@ref), the left-edge of the bin that the normalized ``Λ``
fell into is returned.

## Performance tips

If you are encoding multiple input vectors, it is more efficient to construct a
[`FirstDifferenceEncoding`](@ref) instance and re-use it:

```julia
minval, maxval = 0, 1
encoding = FirstDifferenceEncoding(minval, maxval; n = 4)
pts = [rand(3) for i = 1:1000]
[encode(encoding, x) for x in pts]
```

[^Azami2016]:
    Azami, H., & Escudero, J. (2016). Amplitude-aware permutation entropy:
    Illustration in spike detection and signal segmentation. Computer methods and
    programs in biomedicine, 128, 40-51.
"""
Base.@kwdef struct FirstDifferenceEncoding <: Encoding
    n::Int = 2
    minval::Real
    maxval::Real
    encoder::RectangularBinEncoding

    function FirstDifferenceEncoding(minval::Real, maxval::Real; n = 2)
        encoder = RectangularBinEncoding(FixedRectangularBinning(0, 1, n + 1))
        new(n, minval, maxval, encoder)
    end
end

function encode(encoding::FirstDifferenceEncoding, x::AbstractVector{<:Real})
    L = length(x)
    Λ = 0.0 # a loop is much faster than using `diff` (which allocates a new vector)
    for i = 2:L
       Λ += abs(x[i] - x[i - 1])
    end
    Λ /= (L - 1)
    Λ_normalized = norm_minmax(Λ, encoding.minval, encoding.maxval)

    # Return an integer from the set {1, 2, …, encoding.n}
    return encode(encoding.encoder, Λ_normalized)
end

function decode(encoding::FirstDifferenceEncoding, ω::Int)
    # Return the left-edge of the bin.
    return decode(encoding.encoder, ω)
end

total_outcomes(encoding::FirstDifferenceEncoding) = total_outcomes(encoding.encoder)
