export SymbolicAmplitudeAwarePermutation, probabilities, genentropy

struct SymbolicAmplitudeAwarePermutation <: PermutationProbabilityEstimator

    function SymbolicAmplitudeAwarePermutation()
        new()
    end
end


"""
    AAPE(x, A::Real = 0.5, m::Int = length(a))

Encode relative amplitude information of the elements of `a`.
- `A = 1` emphasizes only average values.
- `A = 0` emphasizes changes in amplitude values.
- `A = 0.5` equally emphasizes average values and changes in the amplitude values.
"""
function AAPE(x, A::Real = 0.5, m::Int = length(x))
    (A/m)*sum(abs.(x)) + (1-A)/(m-1)*sum(abs.(diff(x)))
end

function probabilities(x::Dataset{m, T}, est::SymbolicAmplitudeAwarePermutation) where {m, T}
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    πs = symbolize(x, SymbolicPermutation()) 
    wts = AAPE.(x.data, m)

    probs(πs, wts, normalize = true)
end

function probabilities(x::AbstractVector{T}, est::SymbolicAmplitudeAwarePermutation;
        m::Int = 3, τ::Int = 1) where {T<:Real}
    
    m >= 2 || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    τs = tuple([τ*i for i = 0:m-1]...)
    emb = genembed(x, τs)
    πs = symbolize(emb, SymbolicPermutation()) 
    wts = AAPE.(emb.data, m)

    probs(πs, wts, normalize = true)
end

function genentropy(x::Dataset{m, T}, est::SymbolicAmplitudeAwarePermutation, α::Real = 1; 
        base = 2) where {m, T}
    
    ps = probabilities(x, est)
    @show minimum(ps)
    genentropy(α, ps; base = base)
end

function genentropy(x::AbstractArray{T}, est::SymbolicAmplitudeAwarePermutation, α::Real = 1; 
        m::Int = 3, τ::Int = 1, base = 2) where {T<:Real}
    
    ps = probabilities(x, est, m = m, τ = τ)
    if minimum(ps) < 0
        @show minimum(ps) 
        @show ps
        @show sum(ps)
    end
    genentropy(α, ps; base = base)
end
