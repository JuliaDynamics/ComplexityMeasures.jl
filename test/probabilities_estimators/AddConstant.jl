using Random
using Test
rng = MersenneTwister(1234)

# The BayesianRegularization estimator is only defined for counting-based `ProbabilitiesEstimators`.

@testset "AddConstant: Counting-based outcome space" begin
    x = rand(rng, 1:10., 100)

    os = [
        UniqueElements(),
        OrdinalPatterns(m = 3),
        Dispersion(),
        CosineSimilarityBinning(),
        ValueBinning(RectangularBinning(3)),
    ]
    @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
        est = AddConstant()
        outcomemodel = os[i]

        ps, Ωobs = probabilities_and_outcomes(est, outcomemodel, x)
        @test ps isa Probabilities
        @test probabilities(est, outcomemodel, x) == ps
        @test outcomes(est, outcomemodel, x) == Ωobs

        ps, Ωall = allprobabilities_and_outcomes(est, outcomemodel, x)
        @test ps isa Probabilities
        @test sort(Ωall) == outcome_space(outcomemodel, x)
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
    @testset "$(typeof(os[i]).name.name)" for i in eachindex(os)
        est = AddConstant()
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

# We don't need to test for non-counting based outcome spaces, because the
# AddConstant constructor prevents us from combining such outcome spaces with
# the estimator.
