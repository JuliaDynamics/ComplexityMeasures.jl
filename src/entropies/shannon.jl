export entropy_shannon

"""
    entropy_shannon(args...; base = MathConstants.e)
Equivalent to `entropy_renyi(args...; base, q = 1)` and provided solely for convenience.
Compute the Shannon entropy, given by
```math
H(p) = - \\sum_i p[i] \\log(p[i])
```
"""
entropy_shannon(args...; base = MathConstants.e) = entropy_renyi(args...; base, q = 1)
