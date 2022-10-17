@testset "Statistical Complexity" begin
    x = rand(1000)
    est = SymbolicPermutation(; m=3, τ=1)
    @test statistical_complexity(x, est) isa Real
    # the complexity should be normalized
    @test 0.0 <= statistical_complexity(x, est) <= 1.0
    # the complexity of a monotonically increasing time series should be zero
    @test statistical_complexity(collect(1:100), est) == 0.0
end
