"""
    entropy_definition_compatibility(e::EntropyDefinition, est::DiffEntropyEst)
Return `true` if the given `est` can estimate the given definition of `e`,
otherwise throw an `ArgumentError`.
"""
entropy_definition_compatibility(::Shannon, est::DiffEntropyEst) = true
function entropy_definition_compatibility(e::EntropyDefinition, est::DiffEntropyEst)
    t = string(nameof(typeof(e)))
    throw(ArgumentError("$t entropy not implemented for $(typeof(est)) estimator"))
end

include("nearest_neighbors/nearest_neighbors.jl")
include("order_statistics/order_statistics.jl")
