using Entropies, Test

@testset "Complexity measures" begin
    testfile("./reverse_dispersion.jl")
    testfile("./approx_entropy.jl")
end
