module Entropies
    include("core.jl")
    include("api.jl")
    include("histogram_estimation.jl")
    include("counting_based/CountOccurrences.jl")
    include("symbolic/symbolic.jl")
    include("binning_based/rectangular/rectangular_estimators.jl")
    include("kerneldensity/kerneldensity.jl")
    include("wavelet/wavelet.jl")
    include("nearest_neighbors/nearest_neighbors.jl")
    include("walkthrough/walkthrough.jl")
    include("dispersion/dispersion_entropy.jl")
end
