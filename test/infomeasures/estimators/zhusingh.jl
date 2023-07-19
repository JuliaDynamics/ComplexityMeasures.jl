using ComplexityMeasures
using Test, LinearAlgebra

# Test internals in addition to end-product, because designing an exact end-product
# test is  a mess due to the neighbor searches. If these top-level tests fail, then
# the issue is probability related to these functions.
nns = StateSpaceSet([[-1, -1], [0, -2], [3, 2.0]])
x = SVector(0.0, 0.0)
dists = ComplexityMeasures.maxdists(x, nns)
vol = ComplexityMeasures.volume_minimal_rect(dists)
ξ = ComplexityMeasures.n_borderpoints(x, nns, dists)
@test vol == 24.0
@test ξ == 2.0

nns = StateSpaceSet([[3, 1], [3, -2], [-5, 1.0]])
x = SVector(0.0, 0.0)
dists = ComplexityMeasures.maxdists(x, nns)
vol = ComplexityMeasures.volume_minimal_rect(dists)
ξ = ComplexityMeasures.n_borderpoints(x, nns, dists)
@test vol == 40.0
@test ξ == 2.0

nns = StateSpaceSet([[-3, 1], [3, 1], [5, -2.0]])
x = SVector(0.0, 0.0)
dists = ComplexityMeasures.maxdists(x, nns)
vol = ComplexityMeasures.volume_minimal_rect(dists)
ξ = ComplexityMeasures.n_borderpoints(x, nns, dists)
@test vol == 40.0
@test ξ == 1.0

# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)
N_base3 = ComplexityMeasures.convert_logunit(N, ℯ, 3)

npts = 1000000
ea = information(ZhuSingh(k = 5), rand(npts))
ea_n3 = information(ZhuSingh(k = 5, base = 3), randn(npts))

@test U - max(0.01, U*0.03) ≤ ea ≤ U + max(0.01, U*0.03)
@test N_base3 * 0.98 ≤ ea_n3 ≤ N_base3 * 1.02
