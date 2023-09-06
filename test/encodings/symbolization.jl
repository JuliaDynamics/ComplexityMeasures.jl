using Test
using ComplexityMeasures
using Random
rng = Xoshiro(1234)


# ----------------------------------------------------------------
# UniqueElements
# ----------------------------------------------------------------
x = randn(rng, 100)
o = UniqueElements()
@test symbolize(o, x) isa Vector{<:Integer}


# ----------------------------------------------------------------
# OrdinalPatterns
# ----------------------------------------------------------------
x = randn(rng, 100)
o = OrdinalPatterns(m = 3)
@test symbolize(o, x) isa Vector{<:Integer}

# ----------------------------------------------------------------
# Dispersion
# ----------------------------------------------------------------
x = randn(rng, 100)
o = Dispersion()
@test symbolize(o, x) isa Vector{<:Integer}

# ----------------------------------------------------------------
# CosineSimilarityBinning
# ----------------------------------------------------------------
x = randn(rng, 100)
o = CosineSimilarityBinning()
@test symbolize(o, x) isa Vector{<:Integer}

# ----------------------------------------------------------------
# ValueBinning{<:FixedRectangularBinning}
# ----------------------------------------------------------------
# `Vector`s
x = randn(rng, 100)
f = FixedRectangularBinning(minimum(x), maximum(x), 3)
o = ValueBinning(f)
@test symbolize(o, x) isa Vector{<:Integer}

# `StateSpaceSet`s
y = StateSpaceSet(rand(rng, 100, 2))
ranges = map(i -> range(0, 1, length=5), tuple(1:2...))
f = FixedRectangularBinning(ranges)
o = ValueBinning(f)
@test symbolize(o, y) isa Vector{<:Integer}

# When the dimensions of the fixed rectangular binning and input data don't match
ranges = (0:0.1:1, range(0, 1; length = 101), range(0, 3.2; step = 0.33))
f = FixedRectangularBinning(ranges)
o = ValueBinning(f)
x = rand(rng, 100)
y = StateSpaceSet(rand(rng, 100, 2))
@test_throws DimensionMismatch symbolize(o, x)
@test_throws DimensionMismatch symbolize(o, y)

# ----------------------------------------------------------------
# ValueBinning{<:RectangularBinning}
# ----------------------------------------------------------------
o = ValueBinning(RectangularBinning(3))

# `Vector`s
x = rand(rng, 100)
@test symbolize(o, x) isa Vector{<:Integer}

# `StateSpaceSet`s
y = StateSpaceSet(rand(rng, 100, 2))
@test symbolize(o, y) isa Vector{<:Integer}
