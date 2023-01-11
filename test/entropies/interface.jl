using ComplexityMeasures, Test
using Random: Xoshiro

@testset "method errors" begin
    x = rand(1000)
    @test_throws MethodError entropy(x, 0.1)
    est = AlizadehArghami() # the AlizadehArghami estimator only works for Shannon entropy
    @test_throws MethodError entropy(Tsallis(), AlizadehArghami(), x)
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
    @test entropy_normalized(s, e2, x) ≈ 1 atol = 1e-3
end