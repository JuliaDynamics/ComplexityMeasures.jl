using ComplexityMeasures, Distances
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
    @test compl isa Real
    @test 0 < compl < 0.02
    # complexity of monotonically ascending signal should be exactly zero
    compl = complexity(c, collect(1:100))
    @test compl == 0.0

    # test the wrapper
    entr, compl = entropy_complexity(c, x)
    @test compl isa Real
    @test entr isa Real
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
end