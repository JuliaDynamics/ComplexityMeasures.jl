using Combinatorics
export SequentialCategoricalEncoding

"""
    SequentialCategoricalEncoding <: Encoding
    SequentialCategoricalEncoding(; symbols, m = 2)

An encoding scheme that [`encode`](@ref)s length-`m` categorical vectors onto integers.

## Description

Given a vector of possible `symbols`, `SequentialCategoricalEncoding` constructs all possible 
length-`m` sequential symbol transitions. 

The input vector `χ` is always treated as categorical, and can have any element type
(but encoding/decoding is faster if `χ` is sortable).

## Example

```julia
encoding = SequentialCategoricalEncoding(symbols = ["hello", "there", "skipper"], m = 2)
julia> encoding = SequentialCategoricalEncoding(symbols = ["hello", "there", "skipper"], m = 2)
SequentialCategoricalEncoding, with 3 fields:
 symbols = ["hello", "there", "skipper"]
 encode_dict = Dict(["there", "skipper"] => 4, ["skipper", "hello"] => 5, ["there", "hello"] => 3, ["hello", "skipper"] => 2, ["skipper", "there"] => 6, ["hello", "there"] => 1)
 decode_dict = Dict(5 => ["skipper", "hello"], 4 => ["there", "skipper"], 6 => ["skipper", "there"], 2 => ["hello", "skipper"], 3 => ["there", "hello"], 1 => ["hello", "there"])
```

We can now use `encoding` to encode and decode transitions:

```julia
julia> decode(encoding, 1)
2-element Vector{String}:
 "hello"
 "there"

julia> encode(encoding, ["hello", "there"])
1

julia> encode(encoding, ["there", "skipper"])
4

julia> decode(encoding, 4)
2-element Vector{String}:
 "there"
 "skipper"
```


"""
struct SequentialCategoricalEncoding{M, V, ED, DD} <: Encoding
    symbols::V
    encode_dict::ED
    decode_dict::DD

    function SequentialCategoricalEncoding(; symbols, m = 2)
        s = unique(symbols) # we don't sort, because that would disallow mixing types
        pgen = permutations(s, m)
        T = eltype(s)
        perms = [SVector{m, T}(p) for p in pgen]

        encode_dict = Dict{eltype(perms), Int}()
        decode_dict = Dict{Int, eltype(perms)}()
        for (i, pᵢ) in enumerate(perms)
            encode_dict[pᵢ] = i
            decode_dict[i] = pᵢ
        end
        S, TED, TDD = typeof(s), typeof(encode_dict), typeof(decode_dict)
        return new{m, S, TED, TDD}(s, encode_dict, decode_dict)
    end
end


# Note: internally, we represent the transitions with `StaticVector`s. However,
# `χ` will in general not be a static vector if the user uses `encode` directly. 
# Therefore, we convert to `StaticVector`. This doesn't allocate, so no need to 
# worry about performance. 
function encode(encoding::SequentialCategoricalEncoding{m}, χ::AbstractVector) where {m}
    if m != length(χ)
        throw(ArgumentError("Transition length `m` and length of input must match! Got `m = $m` and `length(χ) = $(length(χ))`"))
    end
    χstatic = SVector{m, eltype(χ)}(χ)
    return encoding.encode_dict[χstatic]
end

function decode(encoding::SequentialCategoricalEncoding{m}, i) where {m}
    return encoding.decode_dict[i]  
end