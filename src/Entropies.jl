module Entropies
    include("core.jl")
    include("histogram_estimation.jl")
    include("counting_based/CountOccurrences.jl")
    include("symbolic/symbolic.jl")
    include("binning_based/rectangular/rectangular_estimators.jl")
    include("wavelet/wavelet.jl")
end
