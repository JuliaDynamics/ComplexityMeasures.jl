# Interface.
testfile("interface.jl")

# Probability estimators.
testfile("estimators/count_occurrences.jl")
testfile("estimators/value_histogram.jl")
testfile("estimators/transfer_operator.jl")
testfile("estimators/naive_kernel.jl")
testfile("estimators/permutation.jl")
testfile("estimators/timescales.jl")
testfile("estimators/dispersion.jl")
testfile("estimators/diversity.jl")

testfile("estimators/spatial/spatial_permutation.jl")
testfile("estimators/spatial/spatial_dispersion.jl")
