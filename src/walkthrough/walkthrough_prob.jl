# The walkthrough probability is not used in itself at the moment, but is included here for
# completeness for the reproduction of Stoop et al. (2021).

function outer_weight(n::Int, 𝐍)
    factorial(BigInt(n)) / prod(factorial.(BigInt.(𝐍)))
end

function inner_weight(n::Int, N::Int, 𝐍, 𝐧ₙ)
    s = length(𝐍)

    denominator_elements = zeros(s)

    for j = 1:s
        # The total number of occurrences for the j-th state.
        Nⱼ = 𝐍[j]

        # The number of times the j-th state has been visited up to position `n`
        nⱼ = 𝐧ₙ[j]

        denominator_elements[j] = factorial(BigInt(Nⱼ - nⱼ))
    end
    return factorial(BigInt(N) - BigInt(n)) / prod(denominator_elements)
end

"""
    walkthrough_prob(x, n::Int, 𝐍, 𝐧)
    walkthrough_prob(x, n::Int, g::EntropyGenerator{WalkthroughEntropy})

The walk-through probability (Stoop et al., 2021)[^Stoop2021] for a symbol sequence `x`
(can be a string, or categorical sequence (e.g. integer vector or `Dataset` of state
vectors).

- `n`: The position within the sequence, where `n ∈ [1, 2, …, N]` and `N` is the total
    number of elements in the sequence.
- `𝐍`: a vector of counts (frequencies) for each unique state in `x`.
- `𝐧`: a vector of vectors, where inner vectors all have `length(unique(x))` elements,
    where `𝐧[i][j]` counts the number of times unique state `j` has appeared in `x[1:i]`.

[^Stoop2021]: Stoop, R. L., Stoop, N., Kanders, K., & Stoop, R. (2021). Excess entropies suggest the physiology of neurons to be primed for higher-level computation. Physical Review Letters, 127(14), 148101.
"""
function walkthrough_prob(x, n::Int, 𝐍, 𝐧)
    𝐧ₙ = 𝐧[n]
    N = length(x)

    𝐏 = 𝐍 ./ N

    # First weight is simple.
    w1 = outer_weight(n, 𝐍)
    w2 = inner_weight(n, N, 𝐍, 𝐧ₙ)

    c2 = [(pⱼ^nⱼ) * w2 for (pⱼ, nⱼ) in zip(𝐏, 𝐧ₙ)]
    c3 = [pⱼ^(𝐧[end][j] - 𝐧[n][j]) for (j, pⱼ) in enumerate(𝐏)]

    return w1 * prod(c2) * prod(c3)
end

function walkthrough_prob(x, n, g::EntropyGenerator{<:WalkthroughEntropy})
    walkthrough_prob(x, n, g.init.𝐍, g.init.𝐧)
end
