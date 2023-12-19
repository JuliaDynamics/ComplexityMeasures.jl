using Test
using ComplexityMeasures

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end
@testset "ComplexityMeasures.jl" begin
    # Core
    testfile("counts/counts.jl")
    testfile("probabilities/probabilities.jl")

    # # Outcome spaces
    testfile("outcome_spaces/outcome_spaces.jl")
    testfile("outcome_spaces/implementations/unique_elements.jl")
    testfile("outcome_spaces/implementations/value_binning.jl")
    testfile("outcome_spaces/implementations/transfer_operator.jl")
    testfile("outcome_spaces/implementations/naive_kernel.jl")
    testfile("outcome_spaces/implementations/permutation.jl")
    testfile("outcome_spaces/implementations/timescales.jl")
    testfile("outcome_spaces/implementations/dispersion.jl")
    testfile("outcome_spaces/implementations/cosine_similarity_binning.jl")
    testfile("outcome_spaces/implementations/spatial/spatial_ordinal_patterns.jl")
    testfile("outcome_spaces/implementations/spatial/spatial_dispersion.jl")

    # # probabilities
    testfile("probabilities_estimators/probabilities_estimators.jl")

    include("infomeasures/infomeasures.jl")
    include("complexity/complexity.jl")

    #When multiscale is exported, this should be turned on
    #include("multiscale/multiscale.jl")

    #Various
    testfile("encodings/encodings.jl")
    testfile("convenience.jl")
    testfile("deprecations.jl")
end
