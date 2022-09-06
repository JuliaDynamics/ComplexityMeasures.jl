export entropy_kozachenkoleonenko

"""
    entropy_kozachenkoleonenko(x::AbstractDataset{D, T}; k::Int = 1, w::Int = 0,
        base::Real = MathConstants.e) where {D, T}

Estimate Shannon entropy to the given `base` using `k`-th nearest neighbor
searches, using the method from Kozachenko & Leonenko (1987)[^KozachenkoLeonenko1987],
as described in Charzyńska and Gambin (2016)[^Charzyńska2016].

`w` is the Theiler window, which determines if temporal neighbors are excluded
during neighbor searches (defaults to `0`, meaning that only the point itself is excluded
when searching for neighbours).

See also: [`entropy_kraskov`](@ref).

[^Charzyńska2016]: Charzyńska, A., & Gambin, A. (2016). Improvement of the k-NN entropy
    estimator with applications in systems biology. Entropy, 18(1), 13.
[^KozachenkoLeonenko1987]: Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of
    the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9-16.
"""
function entropy_kozachenkoleonenko(x::AbstractDataset{D, T}; k::Int = 2, w::Int = 0,
        base::Real = MathConstants.e) where {D, T}

    N = length(x)
    ρs = maximum_neighbor_distances(x, w, k)
    h = D/N*sum(log.(base, ρs)) + log(base, ball_volume(D)) +
        MathConstants.eulergamma + log(base, N - 1)
    return h
end
