using Entropies, Test

@testset begin "Reverse dispersion entropy"
    x = rand(100)
    @test reverse_dispersion(x) isa Real
    @test 0.0 <= reverse_dispersion(x, normalize = true) <= 1.0
end
