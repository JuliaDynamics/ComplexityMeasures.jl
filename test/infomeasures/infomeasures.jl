# Interface.
testfile("interface.jl")

# InformationMeasureDefinition types.
testfile("infomeasure_types/renyi.jl")
testfile("infomeasure_types/shannon.jl")
testfile("infomeasure_types/tsallis.jl")
testfile("infomeasure_types/curado.jl")
testfile("infomeasure_types/stretched_exponential.jl")
testfile("infomeasure_types/kaniadakis.jl")
testfile("infomeasure_types/identification.jl")

include("infomeasure_types/shannon_extropy.jl")
include("infomeasure_types/tsallis_extropy.jl")
include("infomeasure_types/renyi_extropy.jl")


# Estimators
testfile("estimators/estimators.jl")
