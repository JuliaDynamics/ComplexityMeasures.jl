
module Entropies
    using Requires

    include("core.jl")
    include("histogram_estimation.jl")
    include("counting_based/CountOccurrences.jl")
    include("symbolic/symbolic.jl")
    include("kerneldensity/kerneldensity.jl")
    include("wavelet/wavelet.jl")
    include("nearest_neighbors/nearest_neighbors.jl")
    include("binning_based/rectangular_estimators.jl")

    function __init__()
        @require Simplices="d5428e67-3037-59ba-9ab1-57a04f0a3b6a" begin
            using .Simplices
            include("binning_based/transferoperator/triangular/exact/SimplexExact.jl")
            include("binning_based/transferoperator/triangular/point/SimplexPoint.jl")
        end
    end
end
