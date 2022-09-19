using Test

@testset "NN - Kraskov" begin
    m = 4
    τ = 1
    τs = tuple([τ*i for i = 0:m-1]...)
    x = rand(250)
    D = genembed(x, τs)

    @test entropy_kraskov(D, k = 3, w = 1) isa Real
end

@testset "NN - KozachenkoLeonenko" begin
    m = 4
    τ = 1
    τs = tuple([τ*i for i = 0:m-1]...)
    x = rand(250)
    D = genembed(x, τs)

    @test entropy_kozachenkoleonenko(D, k = 3, w = 1) isa Real
end
