export OrdinalPattern

"""
    OrdinalPattern(m = 3, τ = 1; lt = est.lt)

A symbolization scheme that converts the input data to ordinal patterns, which are then
encoded to integers using [`encode_motif`](@ref).

# Usage

    symbolize(x, scheme::OrdinalPattern) → Vector{Int}
    symbolize!(s, x, scheme::OrdinalPattern) → Vector{Int}

If applied to an `m`-dimensional [`Dataset`](@ref) `x`, then `m` and `τ` are ignored,
and each `m`-dimensional `xᵢ ∈ x` is directly encoded using [`encode_motif`](@ref).
Optionally, symbols can be written directly into a pre-allocated integer vector `s`, where
`length(s) == length(x)` using `symbolize!`.

If applied to a univariate vector `x`, then `x` is first converted to a delay
reconstruction using embedding dimension `m` and lag `τ`. The resulting
`m`-dimensional `xᵢ ∈ x` are then encoded using [`encode_motif`](@ref).
If using the in-place variant with univariate input, `s` must obey
`length(s) == length(x)-(est.m-1)*est.τ`.

!!! note
    `OrdinalPattern` is intended for symbolizing *time series*. If providing a short vector,
    say `x = [2, 5, 2, 1, 3, 4]`, then `symbolize(x, OrdinalPattern(m = 2, τ = 1)` will
    first embed `x`, then encode/symbolize each resulting *state vector*, not the original
    input. For symbolizing a permutation pattern, use [`encode_motif`](@ref).

## Examples

```julia
using DelayEmbeddings, Entropies
D = Dataset([rand(7) for i = 1:1000])
s = symbolize(D, OrdinalPattern())
```

See also: [`symbolize`](@ref).
"""
Base.@kwdef struct OrdinalPattern <: SymbolizationScheme
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

function symbolize(x::AbstractDataset{m, T}, scheme::OrdinalPattern) where {m, T}
    m >= 2 || error("Data must be at least 2-dimensional to symbolize. If data is a univariate time series, embed it using `genembed` first.")
    s = zeros(Int, length(x))
    symbolize!(s, x, scheme)
    return s
end

function symbolize(x::AbstractVector{T}, scheme::OrdinalPattern) where {T}
    τs = tuple([scheme.τ*i for i = 0:scheme.m-1]...)
    x_emb = genembed(x, τs)

    s = zeros(Int, length(x_emb))
    symbolize!(s, x_emb, scheme)
    return s
end

function symbolize!(s::AbstractVector{Int}, x::AbstractDataset{m, T},
        scheme::OrdinalPattern) where {m, T}

    @assert length(s) == length(x)

    sp = zeros(Int, m) # pre-allocate a single-symbol vector that can be overwritten.
    fill_symbolvector!(s, x, sp, m, lt = scheme.lt)

    return s
end

function symbolize!(s::AbstractVector{Int}, x::AbstractVector{T},
        scheme::OrdinalPattern) where T

    τs = tuple([scheme.τ*i for i = 0:scheme.m-1]...)
    x_emb = genembed(x, τs)
    symbolize!(s, x_emb, scheme)
end
