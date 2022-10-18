using Distances

@testset "Statistical Complexity" begin

    m, τ = 6, 1
    est = SymbolicPermutation(; m, τ)
    c = StatisticalComplexity(; est, entropy=Renyi())
    # test minimum and maximum complexity entropy curves
    h, c_js = minimum_complexity_entropy(c; num=10000)
    @test minimum(h) == 0
    @test maximum(h) ≈ 1
    @test minimum(c_js) == 0
    # this value is calculated with statcomp
    @test maximum(c_js) ≈ 0.197402387702839

    h, c_js = maximum_complexity_entropy(c)
    @test minimum(h) == 0
    @test 0.99 <= maximum(h) <= 1
    @test minimum(c_js) == 0
    # this value is calculated with statcomp
    @test maximum(c_js) ≈ 0.496700423446187

    x = randn(1000)
    @test complexity(c, x) isa Real
    # the complexity should be normalized
    @test 0.0 <= complexity(c, x) <= maximum(c_js)
    # the complexity of a monotonically increasing time series should be zero
    @test complexity(c, collect(1:100)) == 0.0
    # check that we also get a sensible value for another distance measure, like the Hellinger Distance
    c = StatisticalComplexity(;est, entropy=Renyi(), distance=HellingerDist())
    h, c_js = maximum_complexity_entropy(c)
    @test 0 <= complexity(c, x) <= maximum(c_js)

    # test also for 2-dimensional data
    x = randn(100,100)
    stencil = [1 1; 1 1]
    est = SpatialSymbolicPermutation(stencil, x, false)
    c = StatisticalComplexity(; est, entropy=Renyi())
    # complexity of this should be pretty small
    @test 0 <= complexity(c, x) < 0.001
end
