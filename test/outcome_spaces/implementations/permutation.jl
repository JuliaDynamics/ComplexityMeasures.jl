using ComplexityMeasures, Test
using Random
using ComplexityMeasures.DelayEmbeddings: embed

@testset "API" begin
    m = 3
    # Since all these estimators initialize an encoding,
    # many of the tests such as whether "is_less" is used
    # are actually done in the ordinal pattern encoding test suite
    for S in (OrdinalPatterns, WeightedOrdinalPatterns, AmplitudeAwareOrdinalPatterns)
        @test S() isa OutcomeSpace
        @test S(lt = Base.isless) isa OutcomeSpace
        @test total_outcomes(S(m = 3)) == factorial(3)
    end
end

@testset "Analytic, symbolic" begin
    o = OrdinalPatterns(m = 3, τ = 1)
    N = 1000
    x = StateSpaceSet(repeat([1.1 2.2 3.3], 10))
    y = StateSpaceSet(rand(Random.MersenneTwister(123), N, 3))

    @testset "direct" begin
        p1 = probabilities(o, x)
        p2 = probabilities(o, y)
        @test sum(vec(p1)) ≈ 1.0
        @test sum(vec(p2)) ≈ 1.0
        @test length(p1) == 1 # analytic
        @test length(p2) == 6 # analytic
    end

    @testset "pre-allocated" begin
        s = zeros(Int, N)
        p2 = probabilities!(s, o, y)
        @test sum(p2) ≈ 1.0
    end

    @testset "vector" begin
        z = y[:, 1]
        w = view(z, 1:length(z)-1)
        p1 = probabilities(o, z)
        p2 = probabilities(o, w)
        @test sum(vec(p1)) ≈ sum(vec(p2)) ≈ 1
    end

end

# TODO: would be nice to have analytic tests for amplitude aware
@testset "Uniform distr." begin
    m = 4
    τ = 1
    x = rand(Random.MersenneTwister(1234), 100_000)
    D = embed(x, m, τ)
    @testset "$(S)" for S in (OrdinalPatterns,
        WeightedOrdinalPatterns, AmplitudeAwareOrdinalPatterns)
        o = S(m = m, τ = τ)
        p1 = probabilities(o, x)
        p2 = probabilities(o, D)
        @test all(p1 .≈ p2)
        @test all(p -> 0.03 < p < 0.05, p2)
    end
end

@testset "outcomes" begin
    # With m = 2, ordinal patterns and frequencies are:
    # [1, 2] => 3
    # [2, 1] => 2
    x = [1, 2, 1, 2, 1, 2]
    @testset "$(S)" for S in (OrdinalPatterns,
        WeightedOrdinalPatterns, AmplitudeAwareOrdinalPatterns)
        # don't randomize in the case of equal values, so use Base.isless
        o = S(m = 2, lt = Base.isless)
        probs, πs = probabilities_and_outcomes(o, x)
        @test πs == SVector{2, Int}.([[1, 2], [2, 1]])
        @test probs == [3/5, 2/5]
        o3 = S(m = 3)
        @test outcome_space(o3) == [
            [1, 2, 3],
            [1, 3, 2],
            [2, 1, 3],
            [2, 3, 1],
            [3, 1, 2],
            [3, 2, 1],
        ]
        @test total_outcomes(o3) == factorial(3)
        @test issorted(outcome_space(o3))
    end
end

@testset "codification for $(S)" for S in (OrdinalPatterns,
    WeightedOrdinalPatterns, AmplitudeAwareOrdinalPatterns)
    # Codification of vector inputs (time series)
    x = rand(30)
    @test codify(S(), x) isa Vector{Int}
    @test_throws ArgumentError codify(S(), StateSpaceSet(x))
end
