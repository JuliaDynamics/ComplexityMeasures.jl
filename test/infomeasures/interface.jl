using ComplexityMeasures, Test
using Random: Xoshiro

@testset "info interface: convenience methods" begin
    probs = Probabilities(rand(3))
    @test information(probs) == information(Shannon(), probs)
    @test information(PlugIn(Shannon()), probs) == information(Shannon(), probs)

    # Maximum likelihood is the default estimator for discrete information measures.
    s = Shannon()
    e1 = PowerSpectrum()
    e2 = OrdinalPatterns(; m = 2)
    x = rand(Xoshiro(1234), 10_000)
    @test information_maximum(PlugIn(s), e2) == information_maximum(s, e2)
    @test information_normalized(PlugIn(s), e1, x) == information_normalized(s, e1, x)

    # entropy wrapper function
    @test entropy(Shannon(MathConstants.e), ValueHistogram(4), x) ==
    information(Shannon(MathConstants.e), ValueHistogram(4), x)

    y = rand(50)
    @test entropy(Jackknife(), ValueHistogram(4), y) ==
    information(Jackknife(), ValueHistogram(4), y)

end

@testset "info interface: errors" begin
    x = rand(1000)
    @test_throws MethodError information(x, 0.1)
    # the AlizadehArghami estimator only works for Shannon entropy
    @test_throws ArgumentError information(Tsallis(), AlizadehArghami(), x) # deprecated
    @test_throws ArgumentError information(AlizadehArghami(Tsallis()), x)

    @test_throws ArgumentError entropy(ShannonExtropy(), OrdinalPatterns(), x)


    # some new measure
    struct SomeNewMeasure <: InformationMeasure end
    @test_throws ErrorException information_maximum(SomeNewMeasure(), 2)
end

@testset "info interface: normalization" begin
    # must test both an estimator with known and unknown outcome
    # space without given the data
    s = Shannon()
    e1 = PowerSpectrum()
    e2 = OrdinalPatterns(; m = 2)
    x = rand(Xoshiro(1234), 10_000)
    # Maximum
    @test_throws ErrorException information_maximum(s, e1)
    @test information_maximum(PlugIn(s), e2) == 1
    @test information_maximum(s, e2) == 1
    @test information_maximum(s, e1, x) > 0
    @test information_maximum(s, e2, x) == 1

    # normalized
    s = Shannon(base = 7) # make sure non-standard base is also tested
    @test information_normalized(PlugIn(s), e1, x) > 0
    @test information_normalized(s, e1, x) > 0
    @test information_normalized(s, e2, x) ≈ 1 atol=1e-3
end
