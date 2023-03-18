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
    (entr_min, compl_min), (entr_max, compl_max) = min_max_complexity_curves(c; num_min=10000)
    @test minimum(entr_min) == 0
    @test maximum(entr_min) ≈ 1
    @test minimum(compl_min) == 0
    # this value is calculated with statcomp (R package)
    @test maximum(compl_min) ≈ 0.197402387702839
    @test minimum(entr_max) == 0
    @test 0.99 <= maximum(entr_max) <= 1
    @test minimum(compl_max) == 0
    # this value is calculated with statcomp
    @test maximum(compl_max) ≈ 0.496700423446187
end