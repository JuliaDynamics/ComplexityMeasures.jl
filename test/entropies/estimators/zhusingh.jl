using Test, Distributions, LinearAlgebra
using DelayEmbeddings: Dataset
using StaticArrays: @SVector

# Test internals in addition to end-product, because designing an exact end-product
# test is  a mess due to the neighbor searches. If these top-level tests fail, then
# the issue is probability related to these functions.
nns = Dataset([[-1, -1], [0, -2], [3, 2.0]])
x = @SVector [0.0, 0.0]
dists = Entropies.maxdists(x, nns)
vol = Entropies.volume_minimal_rect(dists)
ξ = Entropies.n_borderpoints(x, nns, dists)
@test vol == 24.0
@test ξ == 2.0

nns = Dataset([[3, 1], [3, -2], [-5, 1.0]])
x = @SVector [0.0, 0.0]
dists = Entropies.maxdists(x, nns)
vol = Entropies.volume_minimal_rect(dists)
ξ = Entropies.n_borderpoints(x, nns, dists)
@test vol == 40.0
@test ξ == 2.0

nns = Dataset([[-3, 1], [3, 1], [5, -2.0]])
x = @SVector [0.0, 0.0]
dists = Entropies.maxdists(x, nns)
vol = Entropies.volume_minimal_rect(dists)
ξ = Entropies.n_borderpoints(x, nns, dists)
@test vol == 40.0
@test ξ == 1.0
using DelayEmbeddings: Dataset

# To ensure minimal rectangle volumes are correct, we also test internals directly here.
# It's not feasible to construct an end-product test due to the neighbor searches.
x = Dataset([[-1, -2], [0, -2], [3, 2]]);
y = Dataset([[3, 1], [-5, 1], [3, -2]]);
@test Entropies.volume_minimal_rect([0, 0], x) == 24
@test Entropies.volume_minimal_rect([0, 0], y) == 40

# -------------------------------------------------------------------------------------
# Check if the estimator converge to true values for some distributions with
# analytically derivable entropy.
# -------------------------------------------------------------------------------------
# Entropy to log with base b of a uniform distribution on [0, 1] = ln(1 - 0)/(ln(b)) = 0
U = 0.00
# Entropy with natural log of 𝒩(0, 1) is 0.5*ln(2π) + 0.5.
N = round(0.5*log(2π) + 0.5, digits = 2)
N_base3 = round((0.5*log(2π) + 0.5) / log(3, ℯ), digits = 2) # custom base

npts = 1000000
ea = entropy(Shannon(; base = 2), Zhu(k = 5), rand(npts))
ea_n = entropy(Shannon(; base = ℯ), Zhu(k = 5), randn(npts))
ea_n3 = entropy(Shannon(; base = 3), Zhu(k = 5), randn(npts))

@test round(ea, digits = 2) == U
@test round(ea_n, digits = 2) == N
@test round(ea_n3, digits = 2) == N_base3

x = rand(1000)
@test_throws ArgumentError entropy(Renyi(q = 2), Zhu(k = 5), x)

# Default is Shannon base-2 differential entropy
est = ZhuSingh()
@test entropy(est, x) == entropy(Shannon(; base = 2), est, x)
