module Entropies 
    include("generalized_entropy.jl")
    
    include("histogram_estimation.jl")
    include("abstract.jl")
    include("symbolic/symbolic.jl")
    include("binning_based/rectangular/rectangular_estimators.jl")
end