using ComplexityMeasures, Distances, DynamicalSystemsBase
using Test
using Random
rng = MersenneTwister(1234)


@testset "Statistical Complexity" begin
    

    m, τ = 6, 1
    x = randn(rng, 10000)
    c = StatisticalComplexity(
        dist=JSDivergence(),
        pest=RelativeAmount(),
        o=OrdinalPatterns(; m, τ),
        hest=Renyi()
    )
    # complexity of white noise should be very close to zero
    compl = complexity(c, x)
    @test 0 < compl < 0.02

    # complexity of monotonically ascending signal should be exactly zero
    compl = complexity(c, collect(1:100))
    @test compl == 0.0

    # test that this also works for another type of entropy
    # we don't have any specific values but we can at least repeat the edge case of noise
    # which should also be close to zeros
    c = StatisticalComplexity(
        dist=JSDivergence(),
        pest=RelativeAmount(),
        o=OrdinalPatterns(; m, τ),
        hest=PlugIn(Tsallis())
    )

    # complexity of white noise should be very close to zero
    compl = complexity(c, x)
    @test 0 < compl < 0.02
    compl = complexity(c, collect(1:100))
    @test compl == 0.0

    # check that error is thrown if we try to call complexity(c, p) with "incomplete" probs vectors
    # we must have empty bins here because total_outcomes = fatorial(6) ≫ 10
    p = probabilities(OrdinalPatterns{m}(τ), randn(10))
    @test_throws ArgumentError complexity(c, p)

    # test the wrapper
    entr, compl = entropy_complexity(c, x)
    @test 0 < compl < 0.02
    @test 0.99 < entr < 1.0

    # test minimum and maximum complexity entropy curves
    min_curve, max_curve = entropy_complexity_curves(c; num_min=10000)
    @test minimum(x[1] for x in min_curve) == 0
    @test maximum(x[1] for x in min_curve) ≈ 1
    @test minimum(x[2] for x in min_curve) == 0
    # this value is calculated with statcomp (R package)
    @test maximum(x[2] for x in min_curve) ≈ 0.197402387702839
    @test minimum(x[1] for x in max_curve) == 0
    @test 0.99 <= maximum(x[1] for x in max_curve) <= 1
    @test minimum(x[2] for x in max_curve) == 0
    # this value is calculated with statcomp
    @test maximum(x[2] for x in max_curve) ≈ 0.496700423446187

    # check that complexity value of schuster map is between minimum and maximum curve
    function schuster(x0=0.5, z=3.0/2)
        return DeterministicIteratedMap(schuster_rule, SVector(x0), [z])
    end
    schuster_rule(x, p, n) = @inbounds SVector((x[1]+x[1]^p[1]) % 1)
    m, τ = 6, 1
    c = StatisticalComplexity(
        dist=JSDivergence(),
        est=OrdinalPatterns(; m, τ),
        entr=Renyi()
    )
    ds = schuster()
    x, t = trajectory(ds, 2^15, Ttr=100)
    entr, compl = entropy_complexity(c, x[:, 1])
    # get indices where the entropy of the system is close to a h-value of the entropy complexity curves
    min_entr_ind = findfirst(isapprox.([x[1] for x in min_curve], 0.5, atol=5e-4) )
    max_entr_ind = findlast(isapprox.([x[1] for x in max_curve], 0.5, atol=5e-3) )
    # get corresponding complexity values
    min_complexity = [x[2] for x in min_curve][min_entr_ind]
    max_complexity = [x[2] for x in max_curve][max_entr_ind]
    @test min_complexity < compl <= max_complexity

    # also test that we get an error if we try an `OutcomeSpace`
    # where the outcome space is not defined a priori
    c = StatisticalComplexity(
        dist=JSDivergence(),
        est=ValueBinning(0.1),
        entr=Tsallis()
    )
    @test_throws ErrorException complexity(c, x)
end


@testset "StatisticalComplexity with extropy" begin
    x = randn(rng, 10000)

    # As with regular entropy, for extropy, the edge case of noise should be close to zeros
    c = StatisticalComplexity(
        dist=JSDivergence(),
        est=OrdinalPatterns(; m, τ),
        entr=TsallisExtropy(q = 5)
    )

    # complexity of white noise should be very close to zero
    compl = complexity(c, x)
    @test 0 < compl < 0.02
    compl = complexity(c, collect(1:100))
    @test compl == 0.0
end


# ----------------------------------------------------------------
# Pretty printing
# ----------------------------------------------------------------
r = repr(StatisticalComplexity())

fns = fieldnames(StatisticalComplexity)
hidden_fields = ComplexityMeasures.hidefields(StatisticalComplexity)
displayed_fields = setdiff(fns, hidden_fields)
for fn in displayed_fields
    @test occursin("$fn = ", r)
end
for fn in hidden_fields
    @test !occursin("entr_val = ", r)
end