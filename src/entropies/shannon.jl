export entropy_shannon
export entropy_shannon_norm

"""
    entropy_shannon(args...; base = MathConstants.e)
Equivalent to `entropy_renyi(args...; base = base, q = 1)` and provided solely for convenience.
Computes the Shannon entropy, given by
```math
H(p) = - \\sum_i p[i] \\log(p[i])
```
"""
entropy_shannon(args...; base = MathConstants.e) =
    entropy_renyi(args...; base = base, q = 1)

"""
    entropy_shannon_norm(args...; base = MathConstants.e)
Equivalent to `entropy_renyi_norm(args...; base = base, q = 1)` and provided solely for convenience.
Computes the normalized Shannon entropy, given by
```math
H(p) = - \\dfrac{\\sum_i p[i] \\log(p[i])}{\\log(N)},
```
where ``N`` is the total number of possible states, as determined by
[`alphabet_length`](@ref).
"""
entropy_shannon_norm(args...; base = MathConstants.e) =
    entropy_renyi_norm(args...; base = base, q = 1)
