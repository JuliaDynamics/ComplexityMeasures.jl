using ComplexityMeasures, Test

@testset "fasthist!" begin
    N = 100
    t = collect(range(1, N; length = N))
    D = StateSpaceSet(reshape(range(1, 3N; length = 3N), (N, 3)))

    for x in (t, D)
        @test ComplexityMeasures.fasthist!(x) isa Vector{Int}
        @test sum(ComplexityMeasures.fasthist!(x)) == length(x) == N
        @test all(isequal(1), ComplexityMeasures.fasthist!(x))
    end
end

@testset "isless_rand" begin
    # because permutations are partially random, we sort many times and check that
    # we get *a* (not *the one*) correct answer every time
    for i = 1:50
        s = sortperm([1, 2, 3, 2], lt = ComplexityMeasures.isless_rand)
        @test s == [1, 2, 4, 3] || s == [1, 4, 2, 3]
    end
end
