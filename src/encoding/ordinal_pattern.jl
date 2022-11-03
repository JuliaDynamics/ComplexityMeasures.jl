export OrdinalPatternEncoding

"""
    OrdinalPatternEncoding <: Encoding
    OrdinalPatternEncoding(m = 3, τ = 1; lt = est.lt)

A encoding scheme that converts the input time series to ordinal patterns, which are
then encoded to integers using [`encode_motif`](@ref), used with
[`outcomes`](@ref).

!!! note
    `OrdinalPatternEncoding` is intended for symbolizing *time series*. If providing a short vector,
    say `x = [2, 5, 2, 1, 3, 4]`, then `outcomes(x, OrdinalPatternEncoding(m = 2, τ = 1)` will
    first embed `x`, then encode/symbolize each resulting *state vector*, not the original
    input. For symbolizing a single vector, use `sortperm` on it and use
    [`encode_motif`](@ref) on the resulting permutation indices.

# Usage

    outcomes(x, scheme::OrdinalPatternEncoding) → Vector{Int}
    outcomes!(s, x, scheme::OrdinalPatternEncoding) → Vector{Int}

If applied to an `m`-dimensional `Dataset` `x`, then `m` and `τ` are ignored,
and `m`-dimensional permutation patterns are obtained directly for each
`xᵢ ∈ x`. Permutation patterns are then encoded as integers using [`encode_motif`](@ref).
Optionally, symbols can be written directly into a pre-allocated integer vector `s`, where
`length(s) == length(x)` using `discretize!`.

If applied to a univariate vector `x`, then `x` is first converted to a delay
reconstruction using embedding dimension `m` and lag `τ`. Permutation patterns are then
computed for each of the resulting `m`-dimensional `xᵢ ∈ x`, and each permutation
is then encoded as an integer using [`encode_motif`](@ref).
If using the in-place variant with univariate input, `s` must obey
`length(s) == length(x)-(est.m-1)*est.τ`.

## Examples

```julia
using DelayEmbeddings, Entropies
D = Dataset([rand(7) for i = 1:1000])
s = outcomes(D, OrdinalPatternEncoding())
```

See also: [`outcomes`](@ref).
"""
Base.@kwdef struct OrdinalPatternEncoding <: Encoding
    m::Int = 3
    τ::Int = 1
    lt::Function = isless_rand
end

function fill_symbolvector!(s, x, sp, m::Int; lt::Function = isless_rand)
    @inbounds for i in eachindex(x)
        sortperm!(sp, x[i], lt = lt)
        s[i] = encode_motif(sp, m)
    end
end

function outcomes(x::AbstractDataset{m, T}, scheme::OrdinalPatternEncoding) where {m, T}
    m >= 2 || error("Data must be at least 2-dimensional to discretize. If data is a univariate time series, embed it using `genembed` first.")
    s = zeros(Int, length(x))
    outcomes!(s, x, scheme)
    return s
end

function outcomes(x::AbstractVector{T}, scheme::OrdinalPatternEncoding) where {T}
    τs = tuple([scheme.τ*i for i = 0:scheme.m-1]...)
    x_emb = genembed(x, τs)

    s = zeros(Int, length(x_emb))
    outcomes!(s, x_emb, scheme)
    return s
end

function outcomes!(s::AbstractVector{Int}, x::AbstractDataset{m, T},
        scheme::OrdinalPatternEncoding) where {m, T}

    @assert length(s) == length(x)

    sp = zeros(Int, m) # pre-allocate a single-symbol vector that can be overwritten.
    fill_symbolvector!(s, x, sp, m, lt = scheme.lt)

    return s
end

function outcomes!(s::AbstractVector{Int}, x::AbstractVector{T},
        scheme::OrdinalPatternEncoding) where T

    τs = tuple([scheme.τ*i for i = 0:scheme.m-1]...)
    x_emb = genembed(x, τs)
    outcomes!(s, x_emb, scheme)
end
