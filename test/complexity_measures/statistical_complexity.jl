using Distances

@testset "Statistical Complexity" begin
    x = randn(1000)
    est = SymbolicPermutation(; m=3, τ=1)
    @test statistical_complexity(x, est) isa Real
    # the complexity should be normalized
    @test 0.0 <= statistical_complexity(x, est) <= 1.0
    # the complexity of a monotonically increasing time series should be zero
    @test statistical_complexity(collect(1:100), est) == 0.0
    # check that we also get a sensible value for another distance measure, like the Hellinger Distance
    @test 0 <= statistical_complexity(x, est, distance=HellingerDistance()) < 1

    # test also for 2-dimensional data
    x = randn(100,100)
    stencil = [1 1; 1 1]
    est = SpatialSymbolicPermutation(stencil, x, false)
    # complexity of this should be pretty small
    @test 0 <= statistical_complexity(x, est) < 0.001
end
