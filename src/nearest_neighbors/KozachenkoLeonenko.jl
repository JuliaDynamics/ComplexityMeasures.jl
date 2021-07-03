export KozachenkoLeonenko, genentropy

"""
    KozachenkoLeonenko(; w::Int = 0) <: EntropyEstimator

Entropy estimator based on nearest neighbors. This implementation is based on Kozachenko
& Leonenko (1987)[^KozachenkoLeonenko1987],
as described in Charzyńska and Gambin (2016)[^Charzyńska2016].

`w` is the Theiler window (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

!!! info
    This estimator is only available for entropy estimation. Probabilities
    cannot be obtained directly.

[^Charzyńska2016]: Charzyńska, A., & Gambin, A. (2016). Improvement of the k-NN entropy estimator with applications in systems biology. Entropy, 18(1), 13.
[^KozachenkoLeonenko1987]: Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9-16.
"""
Base.@kwdef struct KozachenkoLeonenko <: NearestNeighborEntropyEstimator
    k::Int = 1
    w::Int = 0
end

function genentropy(x::AbstractDataset{D, T}, est::KozachenkoLeonenko; base::Real = MathConstants.e) where {D, T}
    N = length(x)
    ρs = maximum_neighbor_distances(x, est)
    h = D/N*sum(log.(base, ρs)) + log(base, ball_volume(D)) +
        MathConstants.eulergamma + log(base, N - 1)
    return h
end
