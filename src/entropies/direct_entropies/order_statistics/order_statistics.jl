"""
    ith_order_statistic(ex, i::Int, n::Int = length(x))

Return the i-th order statistic from the order statistics `ex`, requiring that
`Xᵢ = X₁` if `i < 1` and `Xᵢ = Xₙ` if `i > n`.
"""
function ith_order_statistic(ex, i::Int, n::Int = length(x))
    if i < 1
        return ex[1]
    elseif i > n
        return ex[n]
    else
        return ex[i]
    end
end

include("Vasicek.jl")
include("Ebrahimi.jl")
include("AlizadehArghami.jl")
