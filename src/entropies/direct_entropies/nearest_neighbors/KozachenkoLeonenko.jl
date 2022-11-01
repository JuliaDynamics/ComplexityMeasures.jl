export KozachenkoLeonenko

"""
    KozachenkoLeonenko <: IndirectEntropy
    KozachenkoLeonenko(; k::Int = 1, w::Int = 1, base = 2)

An indirect entropy estimator used in [`entropy`](@ref)`(KozachenkoLeonenko(), x)` to
estimate the Shannon entropy of `x` (a multi-dimensional `Dataset`) to the given
`base` using nearest neighbor searches using the method from Kozachenko &
Leonenko[^KozachenkoLeonenko1987], as described in Charzyńska and Gambin[^Charzyńska2016].

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

In contrast to [`Kraskov`](@ref), this estimator uses only the *closest* neighbor.

[^Charzyńska2016]: Charzyńska, A., & Gambin, A. (2016). Improvement of the k-NN entropy
    estimator with applications in systems biology. Entropy, 18(1), 13.
[^KozachenkoLeonenko1987]: Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of
    the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9-16.
"""
@Base.kwdef struct KozachenkoLeonenko{B} <: IndirectEntropy
    w::Int = 1
    base::B = 2
end

function entropy(e::KozachenkoLeonenko, x::AbstractDataset{D, T}) where {D, T}
    (; w, base) = e
    N = length(x)
    ρs = maximum_neighbor_distances(x, w, 1)
    h = D/N*sum(log.(base, ρs)) + log(base, ball_volume(D)) +
        MathConstants.eulergamma + log(base, N - 1)
    return h
end
