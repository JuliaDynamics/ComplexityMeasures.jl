using ComplexityMeasures
using Test
using Random
rng = Xoshiro(1234)

@testset "Common literature names" begin
    x = randn(rng, 1000)
    @test entropy_distribution(x) == information(SequentialPairDistances(x), x)
    @test entropy_permutation(x) == information(OrdinalPatterns(), x)
    @test entropy_wavelet(x) == information(WaveletOverlap(), x)
    @test entropy_dispersion(x) == information(Dispersion(), x)
    @test entropy_sample(x) == complexity_normalized(SampleEntropy(x), x)
    @test entropy_approx(x) == complexity(ApproximateEntropy(x), x)
end

@testset "probabilities(x)" begin
    x = [1, 1, 2, 2, 3, 3]
    @test probabilities(x) == probabilities(UniqueElements(), x) == [1/3, 1/3, 1/3]
end

@testset "entropy convenience function" begin
    p = Probabilities(rand(rng, 100))
    x = rand(rng, 100)
    o = OrdinalPatterns{3}()
    pest = RelativeAmount()
    e = Shannon()
    hest = Jackknife(e)

    @test entropy(p) == information(Shannon(), p)
    @test entropy(o, x) == information(o, x)
    @test entropy(pest, o, x) == information(pest, o, x) == information(PlugIn(Shannon()), o, x)
    @test entropy(e, pest, o, x) == information(PlugIn(e), pest, o, x)
    @test entropy(hest, pest, o, x) == information(hest, pest, o, x)
end