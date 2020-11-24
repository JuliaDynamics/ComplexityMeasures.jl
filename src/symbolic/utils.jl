""" 
    encode_motif(x, m::Int = length(x)) → s::Int

Encode the length-`m` motif `x` (a vector of indices that would sort some vector `v` 
in ascending order) into its unique integer symbol ``s \\in \\{1, 2, \\ldots, m - 1 \\}``, 
using Algorithm 1 in Berger et al. (2019)[^Berger2019]. 

## Example 

```julia
v = rand(5)

# The indices that would sort `v` in ascending order. This is now a permutation 
# of the index permutation (1, 2, ..., 5)
x = sortperm(v)

# Encode this permutation as an integer.
encode_motif(x)
```
[^Berger2019]: Berger, Sebastian, et al. "Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code." Entropy 21.10 (2019): 1023.
"""
function encode_motif(x, m::Int = length(x))
    n = 0
    for i = 1:m-1
        for j = i+1:m
            n += x[i] > x[j] ? 1 : 0
        end
        n = (m-i)*n
    end
    
    return n
end

function isless_rand(a, b)
    if a == b
        rand(Bool)
    elseif a < b
        true
    else
        false
    end
end

""" Compute probabilities of symbols `Π`, given weights `wts`. """
function probs(Π::AbstractVector, wts::AbstractVector; normalize = true)
    length(Π) == length(wts) || error("Need length(Π) == length(wts)")
    N = length(Π)
    idxs = sortperm(Π)
    sΠ = Π[idxs]   # sorted symbols
    sw = wts[idxs] # sorted weights

    i = 1   # symbol counter
    W = 0.0 # Initialize weight
    ps = Float64[]

    prev_sym = sΠ[1]

    while i <= length(sΠ)
        symᵢ = sΠ[i]
        wtᵢ = sw[i]
        if symᵢ == prev_sym
            W += wtᵢ
        else
            # Finished counting weights for the previous symbol, so push
            # the summed weights (normalization happens later).
            push!(ps, W)

            # We are at a new symbol, so refresh sum with the first weight
            # of the new symbol.
            W = wtᵢ
        end
        prev_sym = symᵢ
        i += 1
    end
    push!(ps, W) # last entry

    # Normalize
    Σ = sum(sw)
    if normalize
        return ps ./ Σ
    else
        return ps
    end
end