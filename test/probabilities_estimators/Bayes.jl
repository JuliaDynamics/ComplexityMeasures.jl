using Random
using Test
rng = MersenneTwister(1234)

# The Bayes estimator is only defined for counting-based `ProbabilitiesEstimators`.

@testset "Counting-based outcome space" begin
    @testset "1D estimators" begin
        x = rand(rng, 1:10., 100)

        os = [
            CountOccurrences(),
            SymbolicPermutation(m = 3),
            Dispersion(),
            Diversity(),
            ValueHistogram(RectangularBinning(3)),
            NaiveKernel(0.1),
        ]
        @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
            est = Bayes(os[i])

            ps, Ωobs = probabilities_and_outcomes(est, x)
            @test ps isa Probabilities
            @test probabilities(est, x) == ps
            @test outcomes(est, x) == Ωobs

            ps, Ωall = allprobabilities_and_outcomes(est, x)
            @test ps isa Probabilities
            @test sort(Ωall) == outcome_space(est, x)
            @test allprobabilities(est, x) isa Probabilities
            @test sort(outcome_space(est, x)) == sort(Ωall)
        end

        # Spatial estimators (all of them are currently counting-based)
        # -------------------------------------------------------------
        x = rand(50, 50)
        os = [
            SpatialDispersion([0 1; 1 0], x, c = 2),
            SpatialSymbolicPermutation([0 1; 1 0], x),
        ]
        @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
            est = Bayes(os[i])
            ps, Ωobs = probabilities_and_outcomes(est, x)
            @test ps isa Probabilities
            @test outcomes(est, x) == Ωobs
            @test probabilities(est, x) == ps

            ps, Ωall = allprobabilities_and_outcomes(est, x)
            @test ps isa Probabilities
            @test sort(Ωall) == sort(outcome_space(est, x))
            @test allprobabilities(est, x) == ps
        end
    end
end

# The Bayes estimator is only defined for counting-based `ProbabilitiesEstimators`.
# An `ArgumentError` should be return for non-counting-compatible `OutcomeSpace`s.
@testset "Non-counting-based outcome spaces" begin
    x = rand(rng, 100)
    os = [
        WaveletOverlap(),
        TransferOperator(RectangularBinning(3)),
        PowerSpectrum(),
        SymbolicAmplitudeAwarePermutation(),
        SymbolicWeightedPermutation(),
    ]
    @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
        est = Bayes(os[i])
        @test_throws ArgumentError probabilities(est, x)
        @test_throws ArgumentError allprobabilities(est, x)
        @test_throws ArgumentError probabilities_and_outcomes(est, x)
        @test_throws ArgumentError allprobabilities_and_outcomes(est, x)
    end
end
