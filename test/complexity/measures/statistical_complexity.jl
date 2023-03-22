using ComplexityMeasures, Distances, DynamicalSystemsBase
using Test

@testset "Statistical Complexity" begin

    m, τ = 6, 1
    x = randn(10000)
    c = StatisticalComplexity(
        dist=JSDivergence(),
        est=SymbolicPermutation(; m, τ),
        entr=Renyi()
    )
    # complexity of white noise should be very close to zero
    compl = complexity(c, x)
    @test 0 < compl < 0.02
    # complexity of monotonically ascending signal should be exactly zero
    compl = complexity(c, collect(1:100))
    @test compl == 0.0

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

    # check that complexity value of logistic map is between minimum and maximum curve
    function logistic(x0=0.4; r = 4.0)
        return DeterministicIteratedMap(logistic_rule, SVector(x0), [r])
    end
    logistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))
    m, τ = 6, 1
    c = StatisticalComplexity(
        dist=JSDivergence(),
        est=SymbolicPermutation(; m, τ),
        entr=Renyi()
    )
    ds = logistic()
    x, t = trajectory(ds, 2^15, Ttr=100)
    entr, compl = entropy_complexity(c, x[:, 1])
    # get indices where the entropy of the system is close to a h-value of the entropy complexity curves
    min_entr_ind = findfirst(isapprox.([x[1] for x in min_curve], 0.5, atol=5e-3) )
    max_entr_ind = findfirst(isapprox.([x[1] for x in max_curve], 0.5, atol=5e-3) )
    # get corresponding complexity values
    min_complexity = [x[2] for x in min_curve][min_entr_ind]
    max_complexity = [x[2] for x in max_curve][max_entr_ind]
    @test min_complexity < compl < max_complexity
end