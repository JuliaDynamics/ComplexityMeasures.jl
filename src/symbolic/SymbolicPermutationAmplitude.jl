export SymbolicPermutationAmplitude, symbolize_amplitudes, symbolize_permamp, entropy

import DelayEmbeddings: Dataset
import StaticArrays: SVector

"""
    SymbolicPermutationAmplitude(m::Int; N::Int = 5, b::Real = 2)

A symbolic permutation probabilities estimator that combines permutation information (with motifs of length `m`)
and absolute magnitude information (discretized into `N` intervals, yielding `N` distinct amplitude symbols).

If the estimator is used for entropy computation, then the entropy is computed 
to base `b` (the default `b = 2` gives the entropy in bits).

The motif length must be ≥ 2. By default `m = 2`, which is the shortest 
possible permutation length which retains any meaningful dynamical information.

[^BandtPompe2002]: Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural complexity measure for time series." Physical review letters 88.17 (2002): 174102.
"""
struct SymbolicPermutationAmplitude <: PermutationProbabilityEstimator
    b::Real
    m::Int
    N::Int
    A::Real
    
    function SymbolicPermutationAmplitude(m::Int; N::Int = 5, A::Real = 0.5, b::Real = 2)
        m >= 2 || throw(ArgumentError("Dimensions of individual marginals must be at least 2. Otherwise, symbol sequences cannot be assigned to the marginals. Got m=$(m)."))

        new(b, m, N, A)
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
    f = (A/m)*sum(abs.(x)) + (1-A)/(m-1)*sum(abs.(diff(x)))
end

function norm_minmax(v; ϵ = 0.001)
    mini, maxi = minimum(v), maximum(v)
    
    if mini == maxi
        l = length(v)
        return [1/l for i in v]
    else
        ϵ .+ (1-2ϵ)*(v .- mini) ./  (maxi - mini)
    end
end

"""
    symbolize_amplitudes(x::Dataset{m, T}, N::Int) where {m, T}

Compute amplitude symbols for each of the elements of `x`, encoding 
the absolute amplitude information within each vector `x` (as opposed
to permutation symbols, where only sorting information is included).
"""
function symbolize_amplitudes(x::Dataset{m, T}, N::Int; A::Real = 0.5) where {m, T}
    Λ = AAPE.(x.data, A)
    Λnorm = norm_minmax(Λ)
    return ceil.(Int, Λnorm ./ (1 / N))
end

"""
    symbolize_permamp(x::Dataset{m, T}, N::Int = 5) where {m, T}

Symbolize the points of `x`, using both permutation and absolute 
amplitude information about the x[i]s. Amplitude information is 
discretized into `N` intervals, yielding `N` possible amplitude
symbols.
"""
function symbolize_permamp(x::Dataset{m, T}, N::Int = 5; A = 0.5) where {m, T}
    0 <= A <= 1 || error("A must be on [0, 1]. Got $(A).")
    permsymbols = symbolize(x, SymbolicPermutation(m))
    ampsymbols = symbolize_amplitudes(x, N, A = A)
    [(x, y) for (x, y) in zip(permsymbols, ampsymbols)]
end

"""
    combine_symbols(πs::AbstractVector{Int}, ϕs::AbstractVector{Int}, 
        m::Int, N::Int)
    combine_symbols!(Φs::AbstractVector{Int}, πs::AbstractVector{Int}, ϕs::AbstractVector{Int}, 
        m::Int, N::Int)

Map permutation symbols `πs` and amplitude symbols `ϕs`, constructed from `m`-dimensional 
state vectors, onto integer symbols `Φs`, without loss of information. 

A pre-allocated integer vector `Φs` may be provided for in-place computations, which 
may be useful for speeding up repeated computations.

## Algorithm

Treat the symbols for the `i`-th state vector, 
``πᵢ \\in [0, 1, \\ldots, M - 1]`` and ``ϕᵢ \\in [1, 2, \\ldots, N]``, as coordinates 
(row, column) in a matrix. We can then  assign a unique integer
``\\Phi_i = \\pi_i N + \\phi``, which counts how many filled rows are filled until 
the row containing `(πᵢ, ϕᵢ)` in encountered (``\\phi_i N``), plus the number of 
remaining elements (``\\phi_i``).
"""
function combine_permamps(πs::AbstractVector{Int}, ϕs::AbstractVector{Int}, m::Int, N::Int)
    M = factorial(m)
    length(πs) == length(ϕs) || error("symbol vectors must be of same length")
    L = length(πs)
    
    Φs = Int[]
    for i = 1:L 
        π, ϕ = πs[i], ϕs[i]
        Φ = π*N + ϕ
        push!(Φs, ϕ)
    end
    
    return Φs
end

function combine_permamps!(Φs::AbstractVector{Int}, πs::AbstractVector{Int}, ϕs::AbstractVector{Int}, m::Int, N::Int)
    M = factorial(m)
    length(πs) == length(ϕs) || error("symbol vectors must be of same length")
    L = length(πs)
    
    for i = 1:L 
        π, ϕ = πs[i], ϕs[i]
        Φ = π*N + ϕ
        push!(Φs, ϕ)
    end
    
    return Φs
end

function symbolize(x::Dataset{m, T}, est::SymbolicPermutationAmplitude) where {m, T}
    0 <= est.A <= 1 || error("A must be on [0, 1]. Got $(est.A).")
    πs = symbolize(x, SymbolicPermutation(m))
    ϕs = symbolize_amplitudes(x, est.N, A = est.A)
    L = length(πs)
    return combine_permamps(πs, ϕs, m, est.N)
end


function probabilities(x::Dataset{m, T}, est::SymbolicPermutationAmplitude, α::Real = 1) where {m, T}
    m == est.m || error("The provided SymbolicPermutationAmplitude($(est.m)) estimator only works for dimension m=$(est.m). Data is $(m)-dimensional. Try providing SymbolicPermutationAmplitude($(m)).")
    syms = symbolize_permamp(x, A = est.A)
    ps = non0hist(syms)
end


function entropy(x::Dataset{m, T}, est::SymbolicPermutationAmplitude, α::Real = 1) where {m, T}
    ps = probabilities(x, est, α)
    
    α < 0 && throw(ArgumentError("Order of generalized entropy must be ≥ 0."))
    if α ≈ 0 # Hartley entropy, max-entropy
        return log(est.b, length(ps)) 
    elseif α ≈ 1
        return -sum( x*log(est.b, x) for x in ps ) #Shannon entropy
    elseif isinf(α)
        return -log(est.b, maximum(ps)) #Min entropy
    else
        return (1/(1-α))*log(est.b, sum(x^α for x in ps) ) #Renyi α entropy
    end
end