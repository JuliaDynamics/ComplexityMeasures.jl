using StaticArrays
export BubbleSortSwapsEncoding
"""
    BubbleSortSwapsEncoding <: Encoding
    BubbleSortSwapsEncoding{m}()

`BubbleSortSwapsEncoding` is used with [`encode`](@ref) to encode a length-`m` input
vector `x` into an integer in the range `ω ∈ 0:((m*(m-1)) ÷ 2)`, by counting the number 
of swaps required for the bubble sort algorithm to  sort `x` in ascending order. 

[`decode`](@ref) is not implemented for this encoding.

## Example

```julia
using ComplexityMeasures
x = [1, 5, 3, 1, 2]
e = BubbleSortSwapsEncoding{5}() # constructor type argument must match length of vector 
encode(e, x)
```
"""
struct BubbleSortSwapsEncoding{m, V <: AbstractVector} <: Encoding
    x::V # tmp vector
end

function BubbleSortSwapsEncoding{m}() where {m}
    if m < 100
        v = zero(MVector{m, eltype(0.0)})
        return BubbleSortSwapsEncoding{m, typeof(v)}(v)
    else
        v = zeros(m)
        return BubbleSortSwapsEncoding{m, typeof(v)}(v)
    end
end

function encode(encoding::BubbleSortSwapsEncoding, x::AbstractVector)
    return n_swaps_for_bubblesort(encoding, x)
end

# super naive bubble sort
function n_swaps_for_bubblesort(encoding::BubbleSortSwapsEncoding, state_vector)
    (; x) = encoding
    x .= state_vector
    L = length(state_vector)
    n = 0
    swapped = true
    while swapped
        swapped = false
        n_swaps = 0
        for j = 1:(L - 1)
            if x[j] > x[j+1]
                n_swaps += 1
                x[j], x[j+1] = x[j+1], x[j] # move smallest element to the right
            end
        end
        if iszero(n_swaps)
            return n
        else
            swapped = true
            n += n_swaps
        end
    end
    return n
end

# there's no meaningful way to define `decode`, so it is not implemented.