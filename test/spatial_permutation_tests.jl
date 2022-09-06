using Entropies, Test

@testset "Spatial Permutation Entropy" begin
    x = [1 2 1; 8 3 4; 6 7 5]
    # Re-create the Ribeiro et al, 2012 using stencil
    # (you get 4 symbols in a 3x3 matrix. For this matrix, the upper left
    # and bottom right are the same symbol. So three probabilities in the end).
    stencil = CartesianIndex.([(1,0), (0,1), (1,1)])
    est = SpatialSymbolicPermutation(stencil, x, false)

    # Generic tests
    ps = probabilities(x, est)
    @test ps isa Probabilities
    @test length(ps) == 3
    @test sort(ps) == [0.25, 0.25, 0.5]

    h = genentropy(x, est, base = 2)
    @test h == 1.5
end