export encode_motif

""" 
    encode_motif(x, m::Int = length(x)) → Int

Encode the length-`m` motif `x` (a vector of indices that would sort some vector `v` in ascending order) 
into its unique integer symbol, using Algorithm 1 in Berger et al. (2019)[^Berger2019].

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