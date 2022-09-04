using Entropies, Test

@testset "Spatiotemporal Permutation Entropy" begin
    x = [1 2 1; 8 3 4; 6 7 5]
    # Re-create the Ribeiro et al, 2012 using stencil instead of specifying m
    stencil = CartesianIndex.([(0,1), (0,1),(1,1)])
    est = SpatiotemporalPermutation(stencil, x, false)

    # Generic tests
    ps = probabilities(x, est)
    @test ps isa Probabilities

    # test case from Ribeiro et al, 2012
    h = genentropy(x, est, base = MathConstants.e)
    @test round(h, digits = 2) ≈ 0.33
end