export binhist

probabilities(x) = _non0hist(x)
function _non0hist(x)
    L = length(x)

    hist = Vector{Float64}()
    # Reserve enough space for histogram:
    sizehint!(hist, L)

    sx = sort(x, alg = QuickSort)

    # Fill the histogram by counting consecutive equal values:
    prev_val, count = sx[1], 0
    for val in sx
        if val == prev_val
            count += 1
        else
            push!(hist, count)
            prev_val = val
            count = 1
        end
    end
    push!(hist, count)

    # Shrink histogram capacity to fit its size:
    sizehint!(hist, length(hist))
    return Probabilities(hist ./ L)
end
_non0hist(x::AbstractDataset) = _non0hist(x.data)
probabilities(x::AbstractDataset) = _non0hist(x.data)


using DelayEmbeddings

# Get both unique elements and their counts
function vec_countmap(x, T = BigInt)
    L = length(x)

    hist = Vector{T}()
    T = eltype(x)
    unique_vals = Vector{T}()

    # Reserve enough space for histogram:
    sizehint!(hist, L)
    sizehint!(unique_vals, L)
    sx = sort(x, alg = QuickSort)

    # Fill the histogram by counting consecutive equal values:
    prev_val, count = sx[1], 0
    push!(unique_vals, sx[1])
    for val in sx
        if val == prev_val
            count += 1
        else
            push!(hist, count)
            push!(unique_vals, val)
            prev_val = val
            count = 1
        end
    end
    push!(hist, count)

    # Shrink histogram capacity to fit its size:
    sizehint!(hist, length(hist))
    sizehint!(unique_vals, length(unique_vals))

    return unique_vals, hist
end

# Get characters to sort them.
vec_countmap(x::AbstractString) = vec_countmap(x |> collect)
vec_countmap(x::AbstractDataset) = vec_countmap(x.data)
