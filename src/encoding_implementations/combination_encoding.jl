export CombinationEncoding

"""
    CombinationEncoding <: Encoding
    CombinationEncoding(encodings)

A `CombinationEncoding` takes multiple [`Encoding`](@ref)s and create a combined
encoding that can be used to encode vectors.

## Encoding/decoding

When used with [`encode`](@ref), each [`Encoding`](@ref) in `encodings` returns
integers in the set `1, 2, …, n_e`, where `n_e` is the total number of outcomes
for a particular encoding. For `k` different encodings, we can thus construct the
cartesian coordinate `(c₁, c₂, …, cₖ)` (`cᵢ ∈ 1, 2, …, n_i`), which can uniquely
be identified by an integer. We can thus identify each unique *combined* encoding
with a single integer.

When used with [`decode`](@ref), the integer symbol is converted to its corresponding
cartesian coordinate, which is used to retrieve the decoded symbols for each of
the encodings.

The total number of outcomes is `sum(total_outcomes(e) for e in encodings)`.

## Examples

```julia
using ComplexityMeasures

# We want to encode the vector `x`.
x = [0.9, 0.2, 0.3]

# To do so, we will use a combination of first-difference encoding, amplitude encoding,
# and ordinal pattern encoding.

encodings = [
    FirstDifferenceEncoding(0, 1; n = 2),
    AmplitudeEncoding(0, 1; n = 5),
    OrdinalPatternEncoding(3) # x is a three-element vector
    ]
c = CombinationEncoding(encodings)

# Encode `x` as integer
ω = encode(c, x)

# Decode symbol (into a vector of decodings, one for each encodings `e ∈ encodings`).
# In this particular case, the first two element will be left-bin edges, and
# the last element will be the decoded ordinal pattern (indices that would sort `x`).
d = decode(c, ω)
```
"""
struct CombinationEncoding{VE, L, C} <: Encoding
    # An iterable of encodings.
    encodings::VE

    # internal fields: LinearIndices/CartesianIndices for encodings/decodings.
    linear_indices::L
    cartesian_indices::C
end

function CombinationEncoding(encodings::Vararg{<:Encoding, N}) where N
    ranges = tuple([1:total_outcomes(e) for e in encodings]...)
    linear_indices = LinearIndices(ranges)
    cartesian_indices = CartesianIndices(ranges)
    VE = typeof(encodings)
    L = typeof(linear_indices)
    C = typeof(cartesian_indices)
    return CombinationEncoding(encodings, linear_indices, cartesian_indices)
end
CombinationEncoding(encodings::Vector{<:Encoding}) = CombinationEncoding(encodings...)

# We could in principle allow any `x` here, but not all encodings support encoding
# single numbers. In particular, the `FirstDifferenceEncoding` isn't even defined
# for single numbers, and `OrdinalPatternEncoding` also isn't defined for single numbers.
# Therefore, we enforce vector-valued input with encoding.
function encode(encoding::CombinationEncoding, x::AbstractVector{<:Real})
    symbols = [encode(e, x) for e in encoding.encodings]
    ω::Int = encoding.linear_indices[symbols...]
    return ω
end

function decode(encoding::CombinationEncoding, ω::Int)
    cidx = encoding.cartesian_indices[ω]
    return [decode(e, cidx[i]) for (i, e) in enumerate(encoding.encodings)]
end

function total_outcomes(encoding::CombinationEncoding)
    return sum(total_outcomes(e) for e in encoding.encodings)
end
