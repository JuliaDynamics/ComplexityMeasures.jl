# This function is part of the DEV API and could be used downstream.
# (documented and tested).
"""
    fasthist!(x) → c::Vector{Int}

Count the occurrences `c` of the unique data values in `x`,
so that `c[i]` is the number of times the value
`sort!(unique(x))[i]` occurs. Hence, this method is useful mostly when
`x` contains integer or categorical data.

Prior to counting, `x` is sorted, so this function also mutates `x`.
Therefore, it is called with `copy` in higher level API when necessary.
This function works for any `x` for which `sort!(x)` works.
"""
function fasthist!(x)
    L = length(x)
    hist = Vector{Int}()
    # Reserve enough space for histogram (Base suggests this improves performance):
    sizehint!(hist, L)
    # Fill the histogram by counting consecutive equal values:
    sort!(x; alg = QuickSort)

    prev_val, count = x[1], 0
    for val in x
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
    return hist
end

# Used for the ordinal pattern outcome spaces where there are no weights.
fasthist!(x, ::Nothing) = fasthist!(x)


"""
    fasthist!(x, weights) → c::Vector{Real}

Similar to `fasthist!(x)`, but here the `weights` are summed up for each unique
entry of `x`. `x` is sorted just like in `fasthist!(x)`.
"""
function fasthist!(x::AbstractVector, weights::AbstractVector{T}) where {T}
    length(x) == length(weights) || error("Need length(x) == length(weights)")

    idxs = sortperm(x)
    x .= x[idxs] # sort in-place
    # weights = weights[idxs] # we don't have to sort them
    L = length(x)

    i = 1
    W = zero(T)
    ps = Vector{T}()
    sizehint!(ps, L)

    prev_sym = first(x)

    @inbounds while i <= L
        symᵢ = x[i]
        wtᵢ = weights[idxs[i]] # get weights at the sorted index
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
    sizehint!(ps, length(ps))
    return ps
end
