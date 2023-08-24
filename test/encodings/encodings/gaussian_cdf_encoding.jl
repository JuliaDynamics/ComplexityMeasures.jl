using Test
using ComplexityMeasures
using Statistics: mean, std
using StaticArrays: SVector

# Convenience constructor.
@test GaussianCDFEncoding(rand(3); μ = 0.0, σ = 0.1) isa GaussianCDFEncoding

# Analytical tests
################################################################
# For a zero-mean Gaussian with c = 3 intervals, zero should map to the symbol 2. This
# should correspond approximately to the interval [1/3, 2/3] when decoded.
encoding = GaussianCDFEncoding(μ = 0.0, σ = 0.1, c = 3)
ezero = encode(encoding, 0.0)
@test ezero == 2

dzero = decode(encoding, ezero) # a two-element static vector
@test dzero isa Vector{<:SVector}
@test first(dzero[1]) ≈ 1/3

# ----------------------------------------------------------------
# Integer encoding.
# ----------------------------------------------------------------
# Li et al. (2018) recommends using at least 1000 data points when estimating
# dispersion entropy.
x = rand(1000)
μ = mean(x)
σ = std(x)
c = 4
m = 4
τ = 1
s = GaussianCDFEncoding(c = c; μ, σ)

# Symbols should be in the set [1, 2, …, c].
symbols = encode.(Ref(s), x)
@test all([s ∈ collect(1:c) for s in symbols])

# Test case from Rostaghi & Azami (2016)'s dispersion entropy paper.
y = [9.0, 8.0, 1.0, 12.0, 5.0, -3.0, 1.5, 8.01, 2.99, 4.0, -1.0, 10.0]
μ = mean(y)
σ = std(y)
encoding = GaussianCDFEncoding(c = 3; μ, σ)
s = encode.(Ref(encoding), y)
s_elwise_paper = [3, 3, 1, 3, 2, 1, 1, 3, 2, 2, 1, 3]
@test s == s_elwise_paper

# ----------------------------------------------------------------
# State vector encoding/decoding (we just re-use the example above)
# ----------------------------------------------------------------
y = [9.0, 8.0, 1.0, 12.0, 5.0, -3.0, 1.5, 8.01, 2.99, 4.0, -1.0, 10.0]
s_elwise_paper = [3, 3, 1, 3, 2, 1, 1, 3, 2, 2, 1, 3]
μ = mean(y)
σ = std(y)
encoding = GaussianCDFEncoding(length(y); c = 3, μ, σ)
s = encode(encoding, y)
@test s isa Int
@test s == encoding.linear_indices[s_elwise_paper...]
@test 1 ≤ s ≤ total_outcomes(encoding)
@test decode(encoding, s) isa AbstractVector{<:SVector}

@test_throws ArgumentError encode(GaussianCDFEncoding(length(y) - 1; c = 3, μ, σ), y)