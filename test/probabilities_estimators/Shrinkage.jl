using Random
using Test
rng = MersenneTwister(1234)
@testset "Shrinkage: counting-based outcome space" begin
    @testset "1D estimators" begin
        x = rand(rng, 1:10., 100)

        os = [
            CountOccurrences(),
            OrdinalPatterns(m = 3),
            Dispersion(),
            Diversity(),
            ValueHistogram(RectangularBinning(3)),
        ]
        @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
            est = Shrinkage(os[i])

            cts, Ωobs = allcounts_and_outcomes(est, x)
            @test sort(Ωobs) == sort(outcome_space(est, x))
            @test length(cts) == total_outcomes(est, x)
            @test allcounts(est, x) == cts

            cts, Ωobs = counts_and_outcomes(est, x)
            @test cts isa Vector{Int}
            @test length(cts) <= total_outcomes(est, x)
            @test length(Ωobs) <= total_outcomes(est, x)
            @test counts(est, x) == cts

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
            SpatialOrdinalPatterns([0 1; 1 0], x),
        ]
        @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
            est = Shrinkage(os[i])
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

# We don't need to test for non-counting based outcome spaces, because the
# Shrinkage constructor prevents us from combining such outcome spaces with
# the estimator.
