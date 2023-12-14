using ComplexityMeasures
using Random
using Test
rng = MersenneTwister(1234)

# The BayesianRegularization estimator is only defined for counting-based outcome spaces.

@testset "BayesianRegularization" begin
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
            est = BayesianRegularization()
            outcomemodel = os[i]

            ps = probabilities(est, outcomemodel, x)
            Ωobs = outcomes(ps)
            @test ps isa Probabilities
            @test probabilities(est, outcomemodel, x) == ps
            @test outcomes(est, outcomemodel, x) == Ωobs

            ps = allprobabilities(est, outcomemodel, x)
            Ωall = outcomes(ps)
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
            est = BayesianRegularization()
            o = os[i]
            ps = probabilities(est, o, x)
            Ωobs = outcomes(ps)
            @test ps isa Probabilities
            @test outcomes(est, o, x) == Ωobs
            @test probabilities(est, o, x) == ps

            ps = allprobabilities(est, o, x)
            Ωall = outcomes(ps)
            @test ps isa Probabilities
            @test sort(Ωall) == sort(outcome_space(est, o, x))
            @test allprobabilities(est, o, x) == ps
        end
    end
end

# We don't need to test for non-counting based outcome spaces, because the
# BayesianRegularization constructor prevents us from combining such outcome spaces with
# the estimator.
