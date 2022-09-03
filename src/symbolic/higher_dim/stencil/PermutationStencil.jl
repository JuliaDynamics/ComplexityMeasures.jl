export PermutationStencil

"""
    PermutationStencil(stencil::AbstractArray{T, N}) <: ProbabilitiesEstimator

Estimate permutation entropies of `N`-dimensional matrices using the provided
`stencil` array, which can be any rectangular shape, and whose elements have
type `T` that can be passed to `Base.isone` (i.e. numbers or booleans).

The size of `stencil` determines the dimension of the rectangular submatrices for
which symbols are generated, and `stencil[i]` determines if the `i`-th
element of the submatrices (using linear indexing) is considered as part of the
ordinal pattern.

# Input requirements

The dimensions `dx, dy, ...` of the `stencil` should be much smaller than the
array to which it is applied. If `nx, ny, ...` are the dimensions of the target
array, it is required that `factorial(dx* dy * ...) << nx * ny * ...`.

## Example

```jldoctest
julia> img = reshape(1:25, 5, 5)
5×5 reshape(::UnitRange{Int64}, 5, 5) with eltype Int64:
 1   6  11  16  21
 2   7  12  17  22
 3   8  13  18  23
 4   9  14  19  24
 5  10  15  20  25

julia> stencil = [1 0 1; 0 1 0; 1 0 1]
3×3 Matrix{Int64}:
 1  0  1
 0  1  0
 1  0  1
```

Then, the upper left `3*3`-submatrix of `img` would be represented by the vector
`[1, 3, 7, 11, 13]`, whose ordinal pattern is converted to a permutation symbol
(integer). Repeated application to all `3*3` submatrices yield a distribution of
integer symbols, from which a probability distribution can be estimated, from
which entropy can be computed.

    PermutationStencil(m::Int, D::Int) <: ProbabilitiesEstimator

Convenience constructor for (hyper)square stencils in dimension `D` where
all elements are true. For the 2D case, this recovers the approach in
Riberio et al. (2012)[^Ribeiro2012].

## Parameter requirements

As for the stencil-based approach, it is required `factorial(d*d) << nx*ny`, where
`ny` and `nx` are the number of rows and columns in the target array.

### Example

The following two approaches are equivalent.

```jldoctest
julia> using Entropies

julia> PermutationStencil(2, 2)
PermutationStencil{2}(Bool[1 1; 1 1])

julia> PermutationStencil([1 1; 1 1])
PermutationStencil{2}(Bool[1 1; 1 1])
```

# Computing entropies

    genentropy(x::AbstractArray{T, 2}, est::SymbolicPermutation2D;
        q = 1, base = MathConstants.e, normalize = true) where T

Compute the generalized order-`q` permutation entropy of a **pre-symbolized** 2D array,
with arbitrary element type `T`. If `normalize == true`, then the entropy is normalized
to the number of possible states.

## Example

```julia
using Entropies
# A pre-symbolized 10000-pixel image, where each pixel now is represented by an integer.
x = rand(1:5, 100, 100)

# Estimate permutation entropy by considering `3 * 3`-sized square pixel blocks,
# using logarithms to base 2
m, N = 3, 2
genentropy(x, PermutationStencil(m, N), base = 2)
```

    entropygenerator(x::AbstractArray{T, 2},
        method::PermutationStencil{2}) → eg::EntropyGenerator

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
m, dim = 2, 2
est = PermutationStencil(m, dim)
eg = entropygenerator(first(images), est)

# The generalized order-`1` (Shannon) normalized permutation entropy to base 2 of each image
[eg(img, base = 2, q = 1, normalize = true) for img in images]
```

# Computing probabilities

    probabilities(x::AbstractArray{T, 2}, est::PermutationStencil{2}) → Probabilities

The same as above, but instead of computing the entropy, directly return the estimated
probabilities.

    probabilitygenerator(x::AbstractArray{T, 2},
        method::PermutationStencil{2}) → ProbabilityGenerator

The same as above, but returns a `ProbabilityGenerator` instead of a `EntropyGenerator`.

[^Ribeiro2012]: Ribeiro, H. V., Zunino, L., Lenzi, E. K., Santoro, P. A., & Mendes, R. S. (2012). Complexity-entropy causality plane as a complexity measure for two-dimensional patterns.
"""
struct PermutationStencil{N} <: ProbabilitiesEstimator
    stencil::AbstractArray{Bool, N}

    function PermutationStencil(s::AbstractArray{Bool, N}) where N
        return new{N}(s)
    end

    function PermutationStencil(s::AbstractArray{T, N}) where {T <: Number, N}
        return new{N}(isone.(s))
    end

    # Square blocks of size `blocksize*blocksize*...` in `N` dimensions.
    function PermutationStencil(blocksize::Int, N::Int)
        return PermutationStencil(ones(Int, tuple(repeat([blocksize], N)...)))
    end
end

function (eg::ProbabilityGenerator{<:PermutationStencil{N}})(; kwargs...) where N
    throw(ArgumentError("Probability generator cannot be called without an argument. Please provide an `$N`-dimensional array for which probabilities should be computed, e.g `probgen(x)`" ))
end

function (eg::EntropyGenerator{<:PermutationStencil{N}})(; kwargs...) where N
    throw(ArgumentError("Entropy generator cannot be called without an argument. Please provide an `$N`-dimensional array for which probabilities should be computed, e.g `probgen(x)`" ))
end

function probabilities(x::AbstractArray{T, N}, est::PermutationStencil{N}) where {T, N<:Integer}
    throw(ArgumentError("Probability estimation using `$N`-dimensional stencils is not yet implemented." ))
end


function genentropy(x::AbstractArray{T, N}, est::PermutationStencil{N}; kwargs...) where {T, N}
    throw(ArgumentError("Entropy estimation using `$N`-dimensional stencils is not yet implemented." ))
end


function entropygenerator(x::AbstractArray{T, N}, method::PermutationStencil{N},
        rng = Random.default_rng()) where {T, N <: Integer}

    pg = probabilitygenerator(x, method, rng)
    init = ()
    return EntropyGenerator(method, pg, x, init, rng)
end

include("stencil_2d.jl")
