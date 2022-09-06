
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
function symprobs(Π::AbstractVector, wts::AbstractVector; normalize = true)
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
