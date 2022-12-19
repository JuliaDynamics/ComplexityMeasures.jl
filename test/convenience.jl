using Entropies
using Test

@testset "Common literature names" begin
    x = randn(1000)
    σ = 0.2

    @testset "Permutation entropy" begin
        @test entropy_permutation(x) == entropy(SymbolicPermutation(), x)
    end

    @testset "Wavelet entropy" begin
        @test entropy_wavelet(x) == entropy(WaveletOverlap(x), x)
    end

    @testset "Dispersion entropy" begin
        @test entropy_dispersion(x) == entropy(Dispersion(), x)
    end

    @testset "Spatial permutation entropy" begin
        x = rand(50, 50)
        stencil = CartesianIndex.([(0,1), (1,1), (1,0)])
        est = SpatialSymbolicPermutation(stencil, x)
        @test entropy_spatial_permutation(x, stencil) == entropy(est, x)
    end
end

@testset "probabilities(x)" begin
    x = [1, 1, 2, 2, 3, 3]
    @test probabilities(x) == probabilities(CountOccurrences(x), x) == [1/3, 1/3, 1/3]
end
