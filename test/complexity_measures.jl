@testset begin "Reverse dispersion entropy"
    x = rand(100)
    @test reverse_dispersion(x) isa Real
end
