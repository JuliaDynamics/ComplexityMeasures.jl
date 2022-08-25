

struct ExcessEntropy <: EntropyMethod end

# TODO: not finished. lacks proper normalization of 𝐧, but the paper is a bit unclear
# about this step.
function entropygenerator(x, method::ExcessEntropy, rng = Random.default_rng())
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
        # ... additional normalized 𝐧 needed for excess entropy.
    )

    return EntropyGenerator(method, x, init, rng)
end


"""
    excess_entropy(x)

Compute the length-normalized excess entropy of `x`.
"""
function excess_entropy(x, n; base = MathConstants.e)
    g = entropygenerator(x, ExcessEntropy())
    return g(n; base = base)
end
