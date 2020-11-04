module Entropies 
    include("histogram_estimation.jl")
    include("generalized_entropy.jl")
    include("abstract.jl")
    include("symbolic/symbolic.jl")
    include("binning_based/rectangular/rectangular_estimators.jl")
end