@testset "Complexity" begin
    testfile("measures/reverse_dispersion.jl")
    testfile("measures/missing_dispersion.jl")
    testfile("measures/entropy_approx.jl")
    testfile("measures/entropy_sample.jl")
    testfile("measures/statistical_complexity.jl")
end
