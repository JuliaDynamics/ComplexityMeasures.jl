@testset "Complexity" begin
    testfile("measures/reverse_dispersion.jl")
    testfile("measures/missing_dispersion.jl")
    testfile("measures/approx_entropy.jl")
    testfile("measures/sample_entropy.jl")
    testfile("multiscale.jl")
end
