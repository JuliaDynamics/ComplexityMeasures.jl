export KozachenkoLeonenko, genentropy

"""
## Nearest neighbour(NN) based

    KozachenkoLeonenko(; w::Int = 1) <: NearestNeighborEntropyEstimator

Entropy estimator based on nearest neighbors. This implementation is based on Kozachenko & Leonenko (1987)[^KozachenkoLeonenko1987],
as described in Charzyńska and Gambin (2016)[^Charzyńska2016].

`w` is the number of nearest neighbors to exclude when searching for neighbours 
(defaults to `0`, meaning that only the point itself is excluded).

!!! info
    This estimator is only available for entropy estimation. Probabilities 
    cannot be obtained directly.

[^Charzyńska2016]: Charzyńska, A., & Gambin, A. (2016). Improvement of the k-NN entropy estimator with applications in systems biology. Entropy, 18(1), 13.
[^KozachenkoLeonenko1987]: Kozachenko, L. F., & Leonenko, N. N. (1987). Sample estimate of the entropy of a random vector. Problemy Peredachi Informatsii, 23(2), 9-16.
"""
struct KozachenkoLeonenko <: NearestNeighborEntropyEstimator
    k::Int
    w::Int
    
    function KozachenkoLeonenko(;k::Int = 1, w::Int = 0)
        new(k, w)
    end
end

function genentropy(x::Dataset{D, T}, est::KozachenkoLeonenko; base::Real = MathConstants.e) where {D, T}
    N = length(x)
    ρs = get_ρs(x, est)
    h = D/N*sum(log.(base, ρs)) + log(base, V(D)) +  MathConstants.eulergamma + log(base, N - 1)
    return h
end

