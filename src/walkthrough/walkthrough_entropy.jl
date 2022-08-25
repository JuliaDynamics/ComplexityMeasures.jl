export walkthrough_entropy
export WalkthroughEntropy

# write into the pre-allocated 𝐧ₙ (a vector containing counts for each unique
# state  up to index n).
function count_occurrences_up_to_n!(𝐧ₙ, n, x, unique_symbols)
    n <= length(x) || throw(ArgumentError("n cannot be larger than length(x)"))

    for i in eachindex(unique_symbols)
        s = unique_symbols[i]
        for j in 1:n
            if (x[j] == s)
                𝐧ₙ[i] += 1
            end
        end
    end
    return 𝐧ₙ
end

function visitations_per_position(x, unique_symbols)
    N = length(x)

    𝐧 = [zeros(BigInt, length(unique_symbols)) for i = 1:N]
    for n in 1:N
        count_occurrences_up_to_n!(𝐧[n], n, x, unique_symbols)
    end

    return 𝐧
end

function entropygenerator(x, method::WalkthroughEntropy, rng = Random.default_rng())
    # 𝐔: unique elements in `x`
    # 𝐍: the corresponding frequencies (𝐍[i] = # times 𝐔[i] occurs).
    𝐔, 𝐍 = vec_countmap(x)

    # 𝐧 is a Vector{Vector{Int}}. 𝐧[i][j] contains the number of times the unique element
    # 𝐔[j] appears in the subsequence `x[1:i]`.
    𝐧 = visitations_per_position(x, 𝐔)

    init = (
        𝐔 = 𝐔,
        𝐍 = 𝐍,
        𝐧 = 𝐧,
    )

    return EntropyGenerator(method, x, init, rng)
end

function conditional_walkprob(n::Int, N::Int, 𝐍, 𝐧)
    if (n == 0)
        return 1.0
    else
        𝐧ₙ = 𝐧[n]
        s = length(𝐍)
        a = prod([binomial(𝐍[j], 𝐧ₙ[j]) for j = 1:s])
        b = binomial(BigInt(N), BigInt(n))
        return a / b
    end
end

# TODO: normalization does nothing at the moment, but is only needed for excess entropy,
# so this doesn't affect the walkthrough entropy. See comment inside function.
function _walkthrough_entropy(n::Int, N::Int, 𝐍, 𝐧; length_normalize = false,
        base = MathConstants.e)

    if (!length_normalize)
        # P(𝐧|𝐍)
        p = conditional_walkprob(n, N, 𝐍, 𝐧)
        return -log(base, p)
    else
        # P(𝐧|𝐍)
        p = conditional_walkprob(n, N, 𝐍, 𝐧)

        # NB: Not sure about the normalization step.
        # Why?
        # Citing from paper: P(𝐧|𝐍) is the multivariate hypergeometric probability
        #  distribution (i.e., for sampling without replacement). It has expectation
        # 𝔼(𝐧) = n𝐩.
        #
        # Length-normalized excess entropy is defined as
        # H(n) = P(𝐧|𝐍) / P(𝔼(𝐧)|𝐍).
        #
        # To compute P(𝔼(𝐧)|𝐍), the elements of 𝐧 must be integers (because the binomial
        # formula is used). However, 𝔼(𝐧) is in general not an integer vector, because 𝐩 is
        # a probability vector, so the integer-vector product n𝐩 yields a vector of floats.
        #
        #𝐄𝐧 = [ceil(Int, StatsBase.mean(nᵢ)) for nᵢ in 𝐧]
        #p𝐄 = conditional_walkprob(n, N, 𝐍, 𝐧)
        -log(base, p)
    end
end

function (eg::EntropyGenerator{<:WalkthroughEntropy})(n::Int;
        length_normalize = false, base = MathConstants.e)

    𝐔, 𝐍, 𝐧 = getfield.(
        Ref(eg.init),
        (:𝐔, :𝐍, :𝐧)
    )

    x = eg.x
    N = length(x)

    0 <= n <= N || throw(ArgumentError("n ∈ [1, 2, …, length(x)] is required. Got n=$n and length(x)=$(length(x))"))

    wte = _walkthrough_entropy(n, N, 𝐍, 𝐧; length_normalize = length_normalize, base = base)
    return convert(Float64, wte)
end


"""
    walkthrough_entropy(x, n::Int; base = MathConstants.e)

Compute the walk-through entropy (Stoop et al., 2021)[^Stoop2021] to the given `base` at
position `n` for a symbol sequence `x`, where `x` can be any categorical iterable.

If computing the walkthrough entropy for multiple `n`, use [`entropygenerator`](@ref) with
    [`WalkthroughEntropy`](@ref).

!!! info
    This estimator is only available for entropy estimation. Probabilities
    cannot be obtained directly.


## Examples

```jldoctest; setup = :(using Entropies)
julia> x = "abc"^10
"abcabcabcabcabcabcabcabcabcabc"

julia> walkthrough_entropy(x, 5)
1.9512293105329133
```

[^Stoop2021]: Stoop, R. L., Stoop, N., Kanders, K., & Stoop, R. (2021). Excess entropies suggest the physiology of neurons to be primed for higher-level computation. Physical Review Letters, 127(14), 148101.
"""
function walkthrough_entropy(x, n::J, base = MathConstants.e) where {J <: Integer}
    g = entropygenerator(x, WalkthroughEntropy())
    # The length-normalized walkthrough entropy is the excess entropy, so here we never
    # normalize.
    return g(n; base = base, length_normalize = false)
end

function walkthrough_entropy(x, n::AbstractVector{J};
        base = MathConstants.e) where {J <: Integer}

    g = entropygenerator(x, WalkthroughEntropy())
    # The length-normalized walkthrough entropy is the excess entropy, so here we never
    # normalize.
    return [g(nᵢ; base = base, length_normalize = false) for nᵢ in n]
end
