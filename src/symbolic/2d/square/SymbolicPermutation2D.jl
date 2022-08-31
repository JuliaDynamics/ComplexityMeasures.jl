export SymbolicPermutation2D

"""
    SymbolicPermutation2D(m::Int = 2)

A symbolic permutation entropy estimator for 2D arrays that symbolizes square sub-matrices
of size `m * m` to estimate probabilities.

The estimator is based on Riberio et al. (2012)[^Ribeiro2012], but is here extended to
generalized entropies of order `q`.

# Parameter requirements

According to Riberio et al. (2012), `m` should be picked such that `(m*m)! << Nc*Nr`,
where `Nc` is the number of columns and `Nr` is the number of rows in the 2D array
for which estimation is targeted.

    genentropy(x::AbstractArray{T, 2}, est::SymbolicPermutation2D;
        q = 1, base = MathConstants.e, normalize = true) where T

Compute the generalized order-`q` permutation entropy of a **pre-symbolized** 2D array,
with arbitrary element type `T`. If `normalize == true`, then the entropy is normalized
to the number of possible states.

## Example

```julia
using Entropies
# A pre-symbolized 10000-pixel image, where each pixel now is represented by an
# integer from 1 to 5.
x = rand(1:5, 100, 100)

# Estimate permutation entropy by considering `3 * 3`-sized square pixel blocks,
# using logarithms to base 2
genentropy(x, SymbolicPermutation2D(m = 3), base = 2)
```

    entropygenerator(x::AbstractArray{T, 2},
        method::SymbolicPermutation2D) → eg::EntropyGenerator{<:SymbolicPermutation2D}

Create an `EntropyGenerator`, using `x` as a template, that efficiently computes the
generalized order-`q` permutation entropy of 2D arrays that has the same size as `x`.

The generator must be called with a 2D array (of same size as `x`) as input, optionally
with keywords `base`, `q` and `normalize` (which have meanings as described above).

## Example

```julia
using Entropies

# 50 images where each of the 10000 pixel take on binary values ("dark" or "light")
images = [rand(["dark", "light"], 100, 100) for i = 1:50]

# Create a generator using the first image as a template. All images are the same size,
# so we can re-use the generator. Use pixel blocks of size `2*2`.
est = SymbolicPermutation2D(m = 2)
eg = entropygenerator(first(images), est)

# The generalized order-`1` (Shannon) normalized permutation entropy to base 2 of each image
[eg(img, base = 2, q = 1, normalize = true) for img in images]
```

[^Ribeiro2012]: Ribeiro, H. V., Zunino, L., Lenzi, E. K., Santoro, P. A., & Mendes, R. S. (2012). Complexity-entropy causality plane as a complexity measure for two-dimensional patterns.
"""
@Base.kwdef struct SymbolicPermutation2D <: ProbabilitiesEstimator
    m::Int = 2
end
