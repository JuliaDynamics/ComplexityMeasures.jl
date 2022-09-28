using Entropies
using Test
import Statistics

x = randn(1000)
σ = 0.2*Statistics.std(x)
# All of the following are API tests.
@test entropy_permutation(x) == entropy(x, SymbolicPermutation())
@test entropy_wavelet(x) == entropy(x, WaveletOverlap())
@test entropy_dispersion(x) == entropy(x, Dispersion())

m = rand(50, 50)
stencil = CartesianIndex.([(0,1), (1,1), (1,0)])
est = SpatialSymbolicPermutation(stencil, m)
@test entropy_spatial_permutation(m, stencil) == entropy(m, est)