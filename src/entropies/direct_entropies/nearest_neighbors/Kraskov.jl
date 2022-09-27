export entropy_kraskov

"""
    Kraskov <: IndirectEntropy
    Kraskov(; k::Int = 1, w::Int = 1, base = 2)

An indirect entropy used in [`entropy`](@ref)`(Kraskov(), x)` to estimate the Shannon
entropy of `x` (a multi-dimensional `Dataset`) to the given
`base` using `k`-th nearest neighbor searches as in [^Kraskov2004].

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

See also: [`KozachenkoLeonenko`](@ref).

[^Kraskov2004]:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information. Physical review E, 69(6), 066138.
"""
struct Kraskov{B}
    k::Int
    w::Int
    base::B
end
Kraskov(; k::Int = 1, w::Int = 1, base = 2) = Kraskov(k, w, base)

function entropy(e::Kraskov, x::AbstractDataset{D, T}) where {D, T}
    (; k, w, base) = e
    N = length(x)
    ρs = maximum_neighbor_distances(x, w, k)
    h = -digamma(k) + digamma(N) + log(base, ball_volume(D)) + D/N*sum(log.(base, ρs))
    return h
end
