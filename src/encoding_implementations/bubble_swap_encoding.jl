using StaticArrays
export BubbleSwapEncoding

struct BubbleSwapEncoding{m, V <: AbstractVector} <: Encoding
    x::V # tmp vector
end

function BubbleSwapEncoding{m}() where {m}
    if m < 100
        v = zero(MVector{m, eltype(0.0)})
        return BubbleSwapEncoding{m, typeof(v)}(v)
    else
        v = zeros(m)
        return BubbleSwapEncoding{m, typeof(v)}(v)
    end
end

function encode(encoding::BubbleSwapEncoding, x::AbstractVector)
    return n_swaps_for_bubblesort(encoding, x)
end

# super naive bubble sort
function n_swaps_for_bubblesort(encoding::BubbleSwapEncoding, state_vector)
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