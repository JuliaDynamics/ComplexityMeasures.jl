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

    # test minimum and maximum complexity entropy curves
    entr, compl = minimum_complexity_entropy(c; num=10000)
    @test minimum(entr) == 0
    @test maximum(entr) ≈ 1
    @test minimum(compl) == 0
    # this value is calculated with statcomp (R package)
    @test maximum(compl) ≈ 0.197402387702839

    entr, compl = maximum_complexity_entropy(c)
    @test minimum(entr) == 0
    @test 0.99 <= maximum(entr) <= 1
    @test minimum(compl) == 0
    # this value is calculated with statcomp
    @test maximum(compl) ≈ 0.496700423446187
end