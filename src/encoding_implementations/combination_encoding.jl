export CombinationEncoding

"""
    CombinationEncoding <: Encoding
    CombinationEncoding(encodings)

A `CombinationEncoding` takes multiple [`Encoding`](@ref)s and creates a combined
encoding that can be used to encode inputs that are compatible with the
given `encodings`.

## Encoding/decoding

When used with [`encode`](@ref), each [`Encoding`](@ref) in `encodings` returns
integers in the set `1, 2, …, n_e`, where `n_e` is the total number of outcomes
for a particular encoding. For `k` different encodings, we can thus construct the
cartesian coordinate `(c₁, c₂, …, cₖ)` (`cᵢ ∈ 1, 2, …, n_i`), which can uniquely
be identified by an integer. We can thus identify each unique *combined* encoding
with a single integer.

When used with [`decode`](@ref), the integer symbol is converted to its corresponding
cartesian coordinate, which is used to retrieve the decoded symbols for each of
the encodings. These decoded symbols are returned as a vector.

The total number of outcomes is `prod(total_outcomes(e) for e in encodings)`.

## Examples

```julia
using ComplexityMeasures

# We want to encode the vector `x`.
x = [0.9, 0.2, 0.3]

# To do so, we will use a combination of first-difference encoding, amplitude encoding,
# and ordinal pattern encoding.

encodings = (
    RelativeFirstDifferenceEncoding(0, 1; n = 2),
    RelativeMeanEncoding(0, 1; n = 5),
    OrdinalPatternEncoding(3) # x is a three-element vector
    )
c = CombinationEncoding(encodings)

# Encode `x` as integer
ω = encode(c, x)

# Decode symbol (into a vector of decodings, one for each encodings `e ∈ encodings`).
# In this particular case, the first two element will be left-bin edges, and
# the last element will be the decoded ordinal pattern (indices that would sort `x`).
d = decode(c, ω)
```
"""
struct CombinationEncoding{E, L, C} <: Encoding
    # An iterable of encodings.
    encodings::E

    # internal fields: LinearIndices/CartesianIndices for encodings/decodings.
    linear_indices::L
    cartesian_indices::C

    function CombinationEncoding(encodings::E, l::L, c::C) where {E, L, C}
        if any(e isa CombinationEncoding for e in encodings)
            s = "CombinationEncoding doesn't accept a CombinationEncoding as one of its " *
             "sub-encodings."
            throw(ArgumentError(s))
        end
        new{E, L, C}(encodings, l, c)
    end
end
CombinationEncoding(encodings) = CombinationEncoding(encodings...)
function CombinationEncoding(encodings::Vararg{<:Encoding, N}) where N
    ranges = tuple([1:total_outcomes(e) for e in encodings]...)
    linear_indices = LinearIndices(ranges)
    cartesian_indices = CartesianIndices(ranges)
    return CombinationEncoding(tuple(encodings...), linear_indices, cartesian_indices)
end

function encode(encoding::CombinationEncoding, χ)
    symbols = CartesianIndex(map(e -> encode(e, χ), encoding.encodings))
    ω::Int = encoding.linear_indices[symbols]
    return ω
end

function decode(encoding::CombinationEncoding, ω::Int)
    cidx = encoding.cartesian_indices[ω]
    return [decode(e, cidx[i]) for (i, e) in enumerate(encoding.encodings)]
end

function total_outcomes(encoding::CombinationEncoding)
    return prod(total_outcomes.(encoding.encodings))
end
