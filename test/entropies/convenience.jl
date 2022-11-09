using Entropies
using Test
import Statistics

x = randn(1000)
σ = 0.2*Statistics.std(x)
# All of the following are API tests.
@test entropy_permutation(x) == entropy(SymbolicPermutation(), x)
@test entropy_wavelet(x) == entropy(WaveletOverlap(), x)
@test entropy_dispersion(x) == entropy(Dispersion(), x)

x = rand(50, 50)
stencil = CartesianIndex.([(0,1), (1,1), (1,0)])
est = SpatialSymbolicPermutation(stencil, x)
@test entropy_spatial_permutation(x, stencil) == entropy(est, x)
