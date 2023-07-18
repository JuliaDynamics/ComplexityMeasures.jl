using ComplexityMeasures, Test
using Random: Xoshiro

@testset "method errors" begin
    x = rand(1000)
    @test_throws MethodError entropy(x, 0.1)
    est = AlizadehArghami() # the AlizadehArghami estimator only works for Shannon entropy
    @test_throws ErrorException entropy(Tsallis(), AlizadehArghami(), x)
end

@testset "normalization" begin
    # must test both an estimator with known and unknown outcome
    # space without given the data
    s = Shannon()
    e1 = PowerSpectrum()
    e2 = SymbolicPermutation(; m = 2)
    x = rand(Xoshiro(1234), 10_000)
    # Maximum
    @test_throws ErrorException entropy_maximum(s, e1)
    @test entropy_maximum(s, e2) == 1
    @test entropy_maximum(s, e1, x) > 0
    @test entropy_maximum(s, e2, x) == 1
    # Normalized
    @test entropy_normalized(s, e1, x) > 0
    @test entropy_normalized(s, e2, x) ≈ 1 atol=1e-3
end

@testset "api" begin
    # Define an estimator for which `entropy_maximum` isn't defined, so we can test
    # that it correctly throws an `ArgumentError`.
    struct SomeEntropy <: EntropyDefinition end
    @test_throws ArgumentError entropy_maximum(SomeEntropy(), 5)
    @test_throws ArgumentError entropy_maximum(MLEntropy(SomeEntropy()), x)

    # DifferentialEntropyEstimators shouldn't work with probabilities
    # There are two conflicting definitions here, because both
    # `entropy(::DifferentialEntropyEstimator, ::AbstractVector)` and
    # `entropy(::DifferentialEntropyEstimator, ::Probabilities)` are defined.
    # probs = Probabilities([0.1, 0.2, 0.3, 0.4])
    # @test_throws ArgumentError entropy(Kraskov(), probs)
end

@testset "convenience" begin
    probs = Probabilities([0.1, 0.2, 0.3, 0.4])

    @test entropy(probs) == entropy(Shannon(), probs)
    @test entropy(MLEntropy(Shannon()), probs) == entropy(Shannon(), probs)

    x = [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.3]
    est = CountOccurrences()
    # Custom base also ensures the last case in `log_with_base` is tested
    @test entropy_normalized(MLEntropy(Shannon(base = 7)), est, x) ==
        entropy_normalized(Shannon(base = 7), est, x)
end
