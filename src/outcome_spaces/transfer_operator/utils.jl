# If x is not sorted, we need to look at all pairwise comparisons
function inds_in_terms_of_unique(x)
    U = unique(x)
    N = length(x)
    Nu = length(U)
    inds = zeros(Int, N)

    for j = 1:N
        xⱼ = view(x, j)
        for i = 1:Nu
            # using views doesn't allocate
            @inbounds if xⱼ == view(U, i)
                inds[j] = i
            end
        end
    end

    return inds,U
end

# Taking advantage of the fact that x is sorted reduces runtime by 1.5 orders of magnitude
# for datasets of >100 000+ points
function inds_in_terms_of_unique_sorted(x) # assumes sorted
    @assert issorted(x)
    N = length(x)
    prev = view(x, 1)
    inds = zeros(Int, N)
    uidx = 1
    @inbounds for j = 1:N
        xⱼ = view(x, j)
        # if the current value has changed, then we know that the corresponding index
        # for the unique point must be incremented by 1
        if xⱼ != prev
            prev = xⱼ
            uidx += 1
        end
        inds[j] = uidx
    end

    return inds
end

function inds_in_terms_of_unique(x, sorted::Bool)
    if sorted
        return inds_in_terms_of_unique_sorted(x)
    else
        return inds_in_terms_of_unique(x)
    end
end

inds_in_terms_of_unique(x::AbstractStateSpaceSet) = inds_in_terms_of_unique(x.data)


function calculate_transition_matrix(S::SparseMatrixCSC;verbose=true)
	S_returned = deepcopy(S)
	calculate_transition_matrix!(S_returned,verbose=verbose)
	return S_returned
end

#normalize each row of S (sum is 1) to get p_ij trans. probabilities
#by looping through CSC sparse matrix efficiently
function calculate_transition_matrix!(S::SparseMatrixCSC;verbose=true)

    stochasticity = true

    St = spzeros(size(S))
    ftranspose!(St,S, x -> x)
    vals = nonzeros(St)
    _,n = size(St)

    #loop over columns
	for j in 1:n
        sumSi = 0.0
        #loop nonzero values from that column
        nzi = nzrange(St,j)
        for i in nzi
            sumSi += vals[i]
        end

        #catch rows (columns) with only zero values
        sumSi == 0.0 && (stochasticity=false)

        #normalize
        for i in nzi
            vals[i] /= sumSi
        end
    end
    ftranspose!(S,St, x->x)
    (stochasticity == false && verbose) && @warn "Transition matrix is not stochastic!"
    nothing
end
