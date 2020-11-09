export ProbabilitiesEstimator, entropy, entropy!, genentropy, Probabilities

struct Probabilities{T}
    p::Vector{T}
    function new(x)
        T = eltype(x)
        s = sum(x)
        if s ≠ 1
            x = x ./ s
        end
        return Probabilities{T}(x)
    end
end


"""
An abstract type for probabilities estimators.
"""
abstract type ProbabilitiesEstimator end


function entropy end
function entropy! end

function probabilities end
function probabilities! end
