export SymbolicAmplitudeAwarePermutation

"""
    SymbolicAmplitudeAwarePermutation(; τ = 1, m = 3, A = 0.5, lt = Entropies.isless_rand) <: PermutationProbabilityEstimator

See docstring for [`SymbolicPermutation`](@ref).
"""
struct SymbolicAmplitudeAwarePermutation{F} <: PermutationProbabilityEstimator
    τ::Int
    m::Int
    A::Float64
    lt::F
end
function SymbolicAmplitudeAwarePermutation(; τ::Int = 1, m::Int = 2, A::Real = 0.5, 
        lt::F = isless_rand) where {F <: Function}
    2 ≤ m || error("Need m ≥ 2, otherwise no dynamical information is encoded in the symbols.")
    0 ≤ A ≤ 1 || error("Weighting factor A must be on interval [0, 1]. Got A=$A.")
    SymbolicAmplitudeAwarePermutation{F}(τ, m, A, lt)
end

"""
    AAPE(x, A::Real = 0.5, m::Int = length(a))

Encode relative amplitude information of the elements of `a`.
- `A = 1` emphasizes only average values.
- `A = 0` emphasizes changes in amplitude values.
- `A = 0.5` equally emphasizes average values and changes in the amplitude values.
"""
function AAPE(x; A::Real = 0.5, m::Int = length(x))
    (A/m)*sum(abs.(x)) + (1-A)/(m-1)*sum(abs.(diff(x)))
end

function probabilities(x::AbstractDataset{m, T}, est::SymbolicAmplitudeAwarePermutation) where {m, T}
    πs = symbolize(x, SymbolicPermutation(m = m, lt = est.lt)) # motif length controlled by dimension of input data
    wts = AAPE.(x.data, A = est.A, m = est.m)

    Probabilities(symprobs(πs, wts, normalize = true))
end

function probabilities(x::AbstractVector{T}, est::SymbolicAmplitudeAwarePermutation) where {T<:Real}
    τs = tuple([est.τ*i for i = 0:est.m-1]...)
    emb = genembed(x, τs)
    πs = symbolize(emb, SymbolicPermutation(m = est.m, lt = est.lt))  # motif length controlled by estimator m
    wts = AAPE.(emb.data, A = est.A, m = est.m)
    p = symprobs(πs, wts, normalize = true)
    Probabilities(p)
end
