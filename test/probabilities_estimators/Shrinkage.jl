using Random
using Test
rng = MersenneTwister(1234)
@testset "Shrinkage: counting-based outcome space" begin
    @testset "1D estimators" begin
        x = rand(rng, 1:10., 100)

        os = [
            UniqueElements(),
            OrdinalPatterns(m = 3),
            Dispersion(),
            CosineSimilarityBinning(),
            ValueBinning(RectangularBinning(3)),
        ]
        @testset "$(nameof(typeof(os[i])))" for i in eachindex(os)
            est = Shrinkage()
            outcomemodel = os[i]

            cts, Ωobs = allcounts_and_outcomes(est, outcomemodel, x)
            @test sort(Ωobs) == sort(outcome_space(est, outcomemodel, x))
            @test length(cts) == total_outcomes(est, outcomemodel, x)
            @test allcounts(est, outcomemodel, x) == cts

            cts, Ωobs = counts_and_outcomes(est, outcomemodel, x)
            @test cts isa Counts
            @test length(cts) <= total_outcomes(est, outcomemodel, x)
            @test length(Ωobs) <= total_outcomes(est, outcomemodel, x)
            @test counts(est, outcomemodel, x) == cts

            ps, Ωobs = probabilities_and_outcomes(est,outcomemodel, x)
            @test ps isa Probabilities
            @test probabilities(est, outcomemodel, x) == ps
            @test outcomes(est, outcomemodel, x) == Ωobs

            ps, Ωall = allprobabilities_and_outcomes(est, outcomemodel, x)
            @test ps isa Probabilities
            @test sort(Ωall) == outcome_space(est, outcomemodel, x)
            @test allprobabilities(est, outcomemodel, x) isa Probabilities
            @test sort(outcome_space(est, outcomemodel, x)) == sort(Ωall)
        end

        # Spatial estimators (all of them are currently counting-based)
        # -------------------------------------------------------------
        x = rand(50, 50)
        os = [
            SpatialDispersion([0 1; 1 0], x, c = 2),
            SpatialOrdinalPatterns([0 1; 1 0], x),
        ]
        @testset "$(nameof(typeof(os[i])))" for i in eachindex(os)
            est = Shrinkage()
            outcomemodel = os[i]
            ps, Ωobs = probabilities_and_outcomes(est, outcomemodel, x)
            @test ps isa Probabilities
            @test outcomes(est, outcomemodel, x) == Ωobs
            @test probabilities(est, outcomemodel, x) == ps

            ps, Ωall = allprobabilities_and_outcomes(est, outcomemodel, x)
            @test ps isa Probabilities
            @test sort(Ωall) == sort(outcome_space(est, outcomemodel, x))
            @test allprobabilities(est, outcomemodel, x) == ps
        end
    end
end

# We don't need to test for non-counting based outcome spaces, because the
# Shrinkage constructor prevents us from combining such outcome spaces with
# the estimator.
