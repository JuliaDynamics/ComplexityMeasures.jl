export entropy_shannon

"""
    entropy_shannon(args...; base = MathConstants.e)
Equivalent with `entropy_renyi(args...; base, q = 1)` and provided solely for convenience.
"""
entropy_shannon(args...; base = MathConstants.e) = entropy_renyi(args...; base, q = 1)
