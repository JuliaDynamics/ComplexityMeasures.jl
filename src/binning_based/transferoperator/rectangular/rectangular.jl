using DelayEmbeddings, SparseArrays

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
    
    return inds
end

# Taking advantage of the fact that x is sorted reduces runtime by 1.5 orders of magnitude 
# for datasets of >100 000+ points
function inds_in_terms_of_unique_sorted(x) # assumes sorted
    @assert issorted(x)
    U = unique(x)
    N, Nu = length(x), length(U)
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

inds_in_terms_of_unique(x::AbstractDataset) = inds_in_terms_of_unique(x.data)

function transopergenerator(pts, method::TransferOperator{<:RectangularBinning})
     
    L = length(pts)
    mini, edgelengths = minima_edgelengths(pts, method.ϵ)

    # The L points visits a total of L bins, which are the following bins: 
    visited_bins = encode_as_bin(pts, mini, edgelengths)
    sort_idxs = sortperm(visited_bins)

    # TODO: fix re-indexing after sorting. Sorting is much faster, so we want to do so.
    #sort!(visited_bins)
    
    # There are N=length(unique(visited_bins)) unique bins.
    # Which of the unqiue bins does each of the L points visit? 
    visits_whichbin = inds_in_terms_of_unique(visited_bins, false) # set to true when sorting is fixed

    # `visitors` lists the indices of the points visiting each of the N unique bins.
    slices = GroupSlices.groupslices(visited_bins)
    visitors = GroupSlices.groupinds(slices) 
    
    # first_visited_by == [x[1] for x in visitors]
    first_visited_by = GroupSlices.firstinds(slices)

    init = (mini = mini,  
        edgelengths = edgelengths,
        visited_bins = visited_bins,
        sort_idxs = sort_idxs,
        visits_whichbin = visits_whichbin,
        visitors = visitors,
        first_visited_by = first_visited_by
        )

    TransferOperatorGenerator(method, pts, init)
end

# """
#     transferoperator(pts::AbstractDataset{D, T}, ϵ::RectangularBinning) → TransferOperatorApproximationRectangular

# Estimate the transfer operator given a set of sequentially ordered points subject to a 
# rectangular partition given by `ϵ`.

# ## Example 

# ```julia
# using DynamicalSystems, Plots, Entropy
# D = 4
# ds = Systems.lorenz96(D; F = 32.0)
# N, dt = 20000, 0.1
# orbit = trajectory(ds, N*dt; dt = dt, Ttr = 10.0)

# # Estimate transfer operator over some coarse graining of the orbit.
# transferoperator(orbit, RectangularBinning(10))
# ```

# See also: [`RectangularBinning`](@ref).
# """
function (tog::TransferOperatorGenerator{T})(;boundary_condition = :circular,
        ) where T <: TransferOperator{<:RectangularBinning};
    
    visitors, 
    first_visited_by, 
    visits_whichbin,
    visited_bins,
    edgelengths,
    mini = getfield.(
        Ref(tog.init), 
            (:visitors, 
            :first_visited_by, 
            :visits_whichbin,
            :visited_bins,
            :edgelengths,
            :mini
            ),
        )

    I = Int32[]
    J = Int32[]
    P = Float64[]

    # Preallocate target index for the case where there is only
    # one point of the orbit visiting a bin.
    target_bin_j::Int = 0
    n_visitsᵢ::Int = 0
    
    if boundary_condition == :circular
        append!(visits_whichbin, [1])
    elseif boundary_condition == :random
        append!(visits_whichbin, [rand(1:length(visits_whichbin))])
    else
        error("Boundary condition $(boundary_condition) not implemented")
    end
    
    # Loop over the visited bins bᵢ
    for i in 1:length(first_visited_by)
        # How many times is this bin visited?
        n_visitsᵢ = length(visitors[i])

        # If both conditions below are true, then there is just one
        # point visiting the i-th bin. If there is only one visiting point and
        # it happens to be the last, we skip it, because we don't know its
        # image.
        if n_visitsᵢ == 1 && !(i == visits_whichbin[end])
            # To which bin does the single point visiting bᵢ jump if we
            # shift it one time step ahead along its orbit?
            target_bin_j = visits_whichbin[visitors[i][1] + 1][1]

            # We now know that exactly one point (the i-th) does the
            # transition from i to the target j.
            push!(I, i)
            push!(J, target_bin_j)
            push!(P, 1.0)
        end

        # If more than one point of the orbit visits the i-th bin, we
        # identify the visiting points and track which bins bⱼ they end up
        # in after the forward linear map of the points.
        if n_visitsᵢ > 1
            timeindices_visiting_pts = visitors[i]

            # TODO: Introduce circular boundary condition. Simply excluding
            # might lead to a cascade of loosing points.

            # If bᵢ is the bin visited by the last point in the orbit, then
            # the last entry of `visiting_pts` will be the time index of the
            # last point of the orbit. In the next time step, that point will
            # have nowhere to go along its orbit (precisely because it is the
            # last data point). Thus, we exclude it.
            if i == visits_whichbin[end]
                #warn("Removing last point")
                n_visitsᵢ = length(timeindices_visiting_pts) - 1
                timeindices_visiting_pts = timeindices_visiting_pts[1:(end - 1)]
            end

            # To which boxes do each of the visitors to bᵢ jump in the next
            # time step?
            target_bins = visits_whichbin[timeindices_visiting_pts .+ 1]
            unique_target_bins = unique(target_bins)

            # Count how many points jump from the i-th bin to each of
            # the unique target bins, and use that to calculate the transition
            # probability from bᵢ to bⱼ.
            #for j in 1:length(unique_target_bins)
            for (j, bᵤ) in enumerate(unique(target_bins))
                n_transitions_i_to_j = sum(target_bins .== bᵤ)

                push!(I, i)
                push!(J, bᵤ)
                push!(P, n_transitions_i_to_j / n_visitsᵢ)
            end
        end
    end
    
    # Transfer operator is just the normalized transition probabilities between the boxes.
    M = sparse(I, J, P)
    
    # Compute the coordinates of the visited bins. bins[i] corresponds to the i-th 
    # row/column of the transfer operator
    unique!(visited_bins)
    bins = [β .* edgelengths .+ mini for β in visited_bins]
    params = (bins = bins, edgelengths = edgelengths)

    TransferOperatorApproximation(tog, M, params)
end

function transferoperator(pts, method::TransferOperator{<:RectangularBinning}; 
        boundary_condition = :circular,
        )
    tog = transopergenerator(pts, method)
    tog(boundary_condition = boundary_condition)
end

"""
    transitioninfo(to::TransferOperatorApproximation) → TransitionInfo
    transitioninfo(to::InvariantMeasure) → TransitionInfo

Convenience method to get the transfer matrix/operator and corresponding bins. Here, 
`bins[i]` corresponds to the i-th row/column of the transfer matrix. Thus, the entry 
`M[i, j]` is the probability of jumping from the state defined by `bins[i]` to the state defined by 
`bins[j]`.

See also: [`TransitionInfo`](@ref), [`TransferOperatorApproximation`](@ref).
"""
function transitioninfo(to::TransferOperatorApproximation{<:TransferOperator{<:RectangularBinning}})
    return TransitionInfo(to.transfermatrix, 
        (bins = to.params.bins, edgelengths = to.params.edgelengths))
end

transitioninfo(iv::InvariantMeasure) = transitioninfo(iv.to)
