using ComplexityMeasures, Test
using Random
using ComplexityMeasures.DelayEmbeddings: embed

@testset "API" begin
    m = 3
    # Since all these estimators initialize an encoding,
    # many of the tests such as whether "is_less" is used
    # are actually done in the ordinal pattern encoding test suite
    for S in (SymbolicPermutation, SymbolicWeightedPermutation, SymbolicAmplitudeAwarePermutation)
        @test S() isa ProbabilitiesEstimator
        @test S(lt = Base.isless) isa ProbabilitiesEstimator
        @test total_outcomes(S(m = 3)) == factorial(3)
    end
end

@testset "Analytic, symbolic" begin
    est = SymbolicPermutation(m = 3, τ = 1)
    N = 1000
    x = Dataset(repeat([1.1 2.2 3.3], 10))
    y = Dataset(rand(Random.MersenneTwister(123), N, 3))

    @testset "direct" begin
        p1 = probabilities(est, x)
        p2 = probabilities(est, y)
        @test sum(vec(p1)) ≈ 1.0
        @test sum(vec(p2)) ≈ 1.0
        @test length(p1) == 1 # analytic
        @test length(p2) == 6 # analytic
    end

    @testset "pre-allocated" begin
        s = zeros(Int, N)
        p2 = probabilities!(s, est, y)
        @test sum(p2) ≈ 1.0
    end

    @testset "vector" begin
        z = y[:, 1]
        w = view(z, 1:length(z)-1)
        p1 = probabilities(est, z)
        p2 = probabilities(est, w)
        @test sum(vec(p1)) ≈ sum(vec(p2)) ≈ 1
    end

end

# TODO: would be nice to have analytic tests for amplitude aware
@testset "Uniform distr." begin
    m = 4
    τ = 1
    x = rand(Random.MersenneTwister(1234), 100_000)
    D = embed(x, m, τ)
    @testset "$(S)" for S in (SymbolicPermutation,
        SymbolicWeightedPermutation, SymbolicAmplitudeAwarePermutation)
        est = S(m = m, τ = τ)
        p1 = probabilities(est, x)
        p2 = probabilities(est, D)
        @test all(p1 .≈ p2)
        @test all(p -> 0.03 < p < 0.05, p2)
    end
end

@testset "outcomes" begin
    # With m = 2, ordinal patterns and frequencies are:
    # [1, 2] => 3
    # [2, 1] => 2
    x = [1, 2, 1, 2, 1, 2]
    @testset "$(S)" for S in (SymbolicPermutation,
        SymbolicWeightedPermutation, SymbolicAmplitudeAwarePermutation)
        # don't randomize in the case of equal values, so use Base.isless
        est = S(m = 2, lt = Base.isless)
        probs, πs = probabilities_and_outcomes(est, x)
        @test πs == SVector{2, Int}.([[1, 2], [2, 1]])
        @test probs == [3/5, 2/5]
        est3 = S(m = 3)
        @test outcome_space(est3) == [
            [1, 2, 3],
            [1, 3, 2],
            [2, 1, 3],
            [2, 3, 1],
            [3, 1, 2],
            [3, 2, 1],
        ]
        @test total_outcomes(est3) == factorial(3)
    end
end