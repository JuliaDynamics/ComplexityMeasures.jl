export ProbabilitiesEstimator, entropy, entropy!, genentropy, Probabilities

struct Probabilities{T}
    p::Vector{T}
    function Probabilities(x)
        T = eltype(x)
        s = sum(x)
        if s ≠ 1
            x = x ./ s
        end
        return new{T}(x)
    end
end


# extend base Vector interface:
for f in (:length, :size, :IteratorSize, :eachindex, :eltype, :lastindex, :firstindex)
    @eval Base.$(f)(d::Probabilities) = $(f)(d.p)
end
@inline Base.iterate(d::Probabilities, i = 1) = iterate(d.p, i)
@inline Base.getindex(d::Probabilities, i) = d.p[i]
@inline Base.:*(d::Probabilities, x::Number) = d.p * x
@inline Base.sum(d::Probabilities{T}) = one(T)

"""
An abstract type for probabilities estimators.
"""
abstract type ProbabilitiesEstimator end


function entropy end
function entropy! end

function probabilities end
function probabilities! end
