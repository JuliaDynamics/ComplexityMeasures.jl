import ComplexityMeasures: Encoding, encode, decode
export UniqueElementsEncoding, encode, decode

# TODO: This should be done using arrays instead. Dicts are slow.
"""
    UniqueElementsEncoding <: Encoding
    UniqueElementsEncoding(x)

`UniqueElementsEncoding` is a generic encoding that encodes each `xᵢ ∈ unique(x)` to one of
the positive integers. The `xᵢ` are encoded according to the order of their first
appearance in the input data.

The constructor requires the input data `x`, since the number of possible symbols
is `length(unique(x))`.

## Example

```julia
using ComplexityMeasures
x = ['a', 2, 5, 2, 5, 'a']
e = UniqueElementsEncoding(x)
encode.(Ref(e), x) == [1, 2, 3, 2, 3, 1] # true
```
"""
struct UniqueElementsEncoding{T, I <: Integer} <: Encoding
    encode_dict::Dict{T, I}
    decode_dict::Dict{I, T}
end
function UniqueElementsEncoding(x)
    # Ecode in order of first appearance, because `sort` doesn't work if we mix types,
    # e.g. `String` and `Int`.
    x_unique = unique(vec(x))
    T = eltype(x_unique)
    encode_dict = Dict{T, Int}()
    decode_dict = Dict{Int, T}()
    for (i, xu) in enumerate(x_unique)
        encode_dict[xu] = i
        decode_dict[i] = xu
    end
    return UniqueElementsEncoding(encode_dict, decode_dict)
end

function UniqueElementsEncoding()
    throw(ArgumentError("`UniqueElementsEncoding` can't be initialized without input data."))
end

function encode(encoding::UniqueElementsEncoding, x)
    return encoding.encode_dict[x]
end

function decode(encoding::UniqueElementsEncoding, ω::I) where I <: Integer
    return encoding.decode_dict[ω]
end
