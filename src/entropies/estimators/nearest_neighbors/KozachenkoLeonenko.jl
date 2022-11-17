export KozachenkoLeonenko

"""
    KozachenkoLeonenko <: EntropyEstimator
    KozachenkoLeonenko(; k::Int = 1, w::Int = 1, base = 2)

The `KozachenkoLeonenko` estimator computes the [`Shannon`](@ref) [`entropy`](@ref) of `x`
(a multi-dimensional `Dataset`) to the given `base`, based on nearest neighbor searches
using the method from Kozachenko & Leonenko (1987)[^KozachenkoLeonenko1987], as described in
Charzyńska and Gambin[^Charzyńska2016].

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

In contrast to [`Kraskov`](@ref), this estimator uses only the *closest* neighbor.

See also: [`entropy`](@ref).

[^Charzyńska2016]: Charzyńska, A., & Gambin, A. (2016). Improvement of the k-NN entropy
    estimator with applications in systems biology. Entropy, 18(1), 13.
[^KozachenkoLeonenko1987]: Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of
    the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9-16.
"""
@Base.kwdef struct KozachenkoLeonenko{B} <: EntropyEstimator
    w::Int = 1
    base::B = 2
end

function entropy(e::KozachenkoLeonenko, x::AbstractDataset{D, T}) where {D, T}
    (; w, base) = e
    N = length(x)
    ρs = maximum_neighbor_distances(x, w, 1)
    # The estimated entropy has "unit" [nats]
    h = 1/N * sum(log.(MathConstants.e, ρs .^ D)) +
        log(MathConstants.e, ball_volume(D)) +
        MathConstants.eulergamma +
        log(MathConstants.e, N - 1)
    return h / log(base, MathConstants.e) # Convert to target unit
end
