import LinearAlgebra: det 
import StaticArrays: SizedMatrix

""" Re-zeros the array `a`. """
function rezero!(a)
    @inbounds for i in eachindex(a)
        a[i] = 0.0
    end
end

""" Fill the elements of vector `from` into vector `into`. """
function fill_into!(into, from)
    @inbounds for i in eachindex(into)
        into[i] = from[i]
    end
end

""" Fill vector `into` at indices `inds` with the elements at `from[inds]`. """
function fill_at_inds!(into, from, inds)
    @inbounds for i in 1:length(inds)
        into[inds[i]] = from[i]
    end
end

function mdet_fm_in_x_view!(M, v, n, starts, stops)
    @inbounds for i = 0:(n - 1)
        M[starts[i+1]:stops[i+1]] = view(v, starts[i+1]:stops[i+1])
    end
    @fastmath det(M)
end


function iscontained!(signs, s_arr, sx, point, dim, temp_arr, starts, stops)
    # Redefine the temporary simplex. This is in-place, so we don't allocate
    # memory. We could also have re-initialised `signs`, but since we're never
    # comparing more than two consecutive signs, this is not necessary.
    rezero!(s_arr)
    rezero!(signs)
    fill_into!(s_arr, sx)

    #Replace first vertex with the point
    fill_at_inds!(s_arr, point, 1:dim)

    # Signed volume
    signs[1] = sign(mdet_fm_in_x_view!(temp_arr, s_arr, dim + 1, starts, stops))

    rezero!(s_arr)
    fill_into!(s_arr, sx) #reset

    for κ = 2:dim # Check remaining signs and stop if sign changes
        # Replace the ith vertex with the point we're cheking (leaving the
        # 1 appended to Vi intact.)
        idxs = ((dim + 1)*(κ - 1)+1):((dim + 1)*(κ - 1)+ 1 + dim - 1)
        fill_at_inds!(s_arr, point, idxs) # ith change

        signs[κ] = sign(mdet_fm_in_x_view!(temp_arr, s_arr, dim + 1, starts, stops))

        if !(signs[κ-1] == signs[κ])
            return false
        end

        rezero!(s_arr)
        fill_into!(s_arr, sx)
    end

    # Last the last vertex with the point in question
    idxs = ((dim + 1)*(dim)+1):((dim+1)^2-1)
    fill_at_inds!(s_arr, point, idxs)

    signs[end] = sign(mdet_fm_in_x_view!(temp_arr, s_arr, dim + 1, starts, stops))

    if !(signs[end-1] == signs[end])
       return false
    else
        return true
    end
end

function innerloop_optim!(inds::Vector{Int}, signs, s_arr, Sj, pt, dim::Int, M, i::Int, temp_arr, starts, stops)
    for j in 1:length(inds)
        if iscontained!(signs, s_arr, Sj[j], pt, dim, temp_arr, starts, stops)
            M[inds[j], i] += 1.0
        end
    end
end

"""
    reshape_simplices(pts::Vector{Vector{T}}, DT::DelaunayTriangulation) where T

Creates alternative representations of the simplices that allow for efficient
(mostly non-allocating) checking if a point lies inside a simplex. 

This function adds some information needed when calculating the orientations
of the simplices *before* we start the computation (appends a row of ones on top of the
matrix representation of the simplex). Otherwise, the fallback is machinery in Simplices.jl,
which is much slower because the additional information must be added
`(n_simplices^2)*n_sample_pts` times (creating that many new matrices), which takes forever.

Reshaping the simplices and appending that information beforehand avoids excessive
memory allocation.
"""
function reshape_simplices(pts, DT::DelaunayTriangulation)
    n_simplices = length(DT.indices)
    n_vertices = length(DT[1])
    dim = n_vertices - 1

    # The [:, j, i] th entry of these two arrays holds the jth vertex of the
    # ith simplex, but instead of having just `dim` vertices, we append a `1`
    # to the end of the vectors. This allows for efficent (non-allocating)
    # computation within the `contains_point_lessalloc!` function. If we instead
    # would have appended the 1's inside that function, we would be performing
    # memory-allocating operations, which are very expensive. Doing this instead
    # gives orders of magnitude speed-ups for sufficiently large triangulations.
    S1 = Array{Float64}(undef, dim + 1, dim + 1, n_simplices)

    # Collect simplices in the form of (dim+1)^2-length column vectors. This
    # also helps with the
    NDIM, N_SIMPLICES = (dim+1)^2, n_simplices
    simplices = SizedMatrix{NDIM}{N_SIMPLICES}(zeros((NDIM, N_SIMPLICES)))

    @inbounds for i in 1:n_simplices
        for j in 1:n_vertices
            S1[:, j, i] = vcat(pts[DT[i][j]], 1.0)
        end

        simplices[:, i] = reshape(S1[:, :, i], (dim+1)^2)
    end

    return simplices
end


"""
idxs_potentially_intersecting_simplices(all_pts::Vector{Vector{T}},
        DT::DelaunayTriangulation, idx::Int) where T

Given a delaunay triangulation `DT` of `all_pts[1:(end - 1)]`, find the indices of
the simplices that potentially intersect with the image simplex with index `idx`.
"""
function idxs_potentially_intersecting_simplices(all_pts, DT::DelaunayTriangulation, idx::Int)
	# Vector that will hold the indices of the simplices potentially
	# intersecting with the image of simplex #idx
	inds_potential_simplices = Int[]

	n_simplices = length(DT)

	original_pts = all_pts[1:(end - 1)]
	forward_pts = all_pts[2:end]

	simplices = [MutableSimplex(original_pts[DT[i]]) for i = 1:n_simplices]
	image_simplices = [MutableSimplex(forward_pts[DT[i]]) for i = 1:n_simplices]

	cs = [centroid(simplices[i]) for i = 1:n_simplices]
	cs_im = [centroid(image_simplices[i]) for i = 1:n_simplices]

	rs = [radius(simplices[i]) for i = 1:n_simplices]
	rs_im = [radius(image_simplices[i]) for i = 1:n_simplices]

	@inbounds for i = 1:n_simplices
	    δ = transpose(cs_im[idx] .- cs[i]) * (cs_im[idx] .- cs[i]) .- ((rs_im[idx] + rs[i])^2)
	    if δ[1] < 0
	        push!(inds_potential_simplices, i)
	    end
	end

	return inds_potential_simplices
end

""" Get the simplices at the given indices. """
function get_simplices_at_inds!(simps, inds::Vector{Int}, simplices)
    for i in 1:length(inds)
        simps[i] = simplices[:, inds[i]]
    end
end