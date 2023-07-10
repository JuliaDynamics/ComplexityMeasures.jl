using Test
using ComplexityMeasures

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end
@testset "ComplexityMeasures.jl" begin
    # Probability estimators
    testfile("probabilities/estimators/count_occurrences.jl")
    testfile("probabilities/estimators/value_histogram.jl")
    testfile("probabilities/estimators/transfer_operator.jl")
    testfile("probabilities/estimators/naive_kernel.jl")
    testfile("probabilities/estimators/permutation.jl")
    testfile("probabilities/estimators/timescales.jl")
    testfile("probabilities/estimators/dispersion.jl")
    testfile("probabilities/estimators/diversity.jl")
    testfile("probabilities/estimators/spatial/spatial_permutation.jl")
    testfile("probabilities/estimators/spatial/spatial_dispersion.jl")
    # probabilities functions
    testfile("probabilities/api.jl")


    include("entropies/entropies.jl")
    include("complexity/complexity.jl")

    # When multiscale is exported, this should be turned on
    # include("multiscale/multiscale.jl")

    # Various
    testfile("utils/fasthist.jl")
    testfile("utils/bin_encoding.jl")
    testfile("utils/encoding.jl")
    testfile("convenience.jl")
    testfile("deprecations.jl")
end
