export disequilibriumgenerator

function probabilitygenerator(x::AbstractArray{T, 2}, method::PermutationStencil{2},
        rng = Random.default_rng()) where T

    ny, nx = size(x)

    # From a submatrix of size `ny`*`nx`, what are the linear indices that select
    # the correct elements, given `method.stencil`?
    stencil = method.stencil
    stencil_submatrix_idxs = LinearIndices(stencil)[findall(stencil .== true)]

    # Picking the first `L` elements of x to infer type from `x`, so that we can
    # use non-numeric types as input too.
    L = count(stencil .== true)
    x̂ = x[1:L]
    x̂_sorted = [i for i = 1:L]

    # Pre-allocate motif vector. Picking sub-matrices means there are fewer
    # than `ny`/`nx` rows/columns
    # to pick from.
    dy, dx = size(method.stencil)
    nr = length(1:nx-dy+1) # rows
    nc = length(1:ny-dx+1) # columns
    motifs = zeros(Int, nr * nc)

    init = (
        x̂ = x̂,
        x̂_sorted = x̂_sorted,
        motifs = motifs,
        stencil_submatrix_idxs = stencil_submatrix_idxs,
        dx = dx,
        dy = dy,
    )
    return ProbabilityGenerator(method, x, init, rng)
end

function update_x̂_from_stencil!(x̂, X::AbstractArray{T, 2}, i, j, dx, dy,
        stencil_submatrix_idxs::Vector{Int}) where T

    Q = view(X, j:j+dy-1, i:i+dx-1)

    k = 1
    for i in stencil_submatrix_idxs
        x̂[k] = Q[i]
        k += 1
    end
end

function update_motifs_from_stencil!(motifs, x̂, x̂_sorted, X::AbstractArray{T, 2}, dx, dy,
        stencil_submatrix_idxs) where T

    ny, nx = size(X)
    rx = 1:nx-dx+1 # columns
    ry = 1:ny-dy+1 # rows
    k = 1
    for i in rx
        for j in ry
            update_x̂_from_stencil!(x̂, X, i, j, dx, dy, stencil_submatrix_idxs)
            sortperm!(x̂_sorted, x̂)
            update_motif!(motifs, k, Entropies.encode_motif(x̂_sorted))
            k += 1
        end
    end
    return motifs
end

function (eg::ProbabilityGenerator{<:PermutationStencil{2}})(X::AbstractArray{T, 2}) where T
    x̂, x̂_sorted, motifs, stencil_submatrix_idxs, dx, dy = getfield.(
        Ref(eg.init),
        (:x̂, :x̂_sorted, :motifs, :stencil_submatrix_idxs, :dx, :dy)
    )
    # Loop through the d*d-sized submatrices and symbolize them, one by one, into
    # the pre-allocated `motifs` vector.
    update_motifs_from_stencil!(motifs, x̂, x̂_sorted, X, dx, dy, stencil_submatrix_idxs)

    return Probabilities(_non0hist(motifs) ./ length(motifs))
end

function probabilities(x::AbstractArray{T, 2},
        est::PermutationStencil{2}) where {T}

    pg = probabilitygenerator(x, est)
    return pg(x)
end

function (eg::EntropyGenerator{<:PermutationStencil{2}})(X::AbstractArray{T, 2};
        base = MathConstants.e, normalize = true, q = 1) where T

    ps = eg.probability_generator(X)
    if normalize
        # For `d > 4`, we need BigInt to compute the factorial of the product.
        dx = eg.probability_generator.init.dx
        dy = eg.probability_generator.init.dy
        nc = convert(Float64, log(base, factorial(BigInt(dx) * BigInt(dy))))
        return float(genentropy(ps, base = base, q = q) / nc)
    else
        return genentropy(ps, base = base, q = q)
    end
end

function entropygenerator(x::AbstractArray{T, 2}, method::PermutationStencil{2},
        rng = Random.default_rng()) where T

    pg = probabilitygenerator(x, method, rng)
    init = ()
    return EntropyGenerator(method, pg, x, init, rng)
end


function genentropy(x::AbstractArray{T, 2}, est::PermutationStencil{2};
        q = 1, base = MathConstants.e, normalize = true) where {T}
    eg = entropygenerator(x, est)
    eg(x, base = base, q = q, normalize = normalize)
end


function _qmax(dx, dy; base = MathConstants.e)
    dxb, dyb = BigInt(dx), BigInt(dy)
    f = factorial(dxb*dyb)
    qmax = -0.5 * (
            (f + 1)/f * log(base, f + 1) -
            2 * log(base, 2 * f) + log(base, f)
        )
    return convert(Float64, qmax)
end

function disequilibrium(x::AbstractArray{T, 2}, est::PermutationStencil{2};
        base = MathConstants.e) where T

    dy, dx = size(est.stencil)
    diseq_gen = disequilibriumgenerator(x, est)
    𝐏 = diseq_gen.probability_generator(x)
    N = length(𝐏)
    𝐏ₑ = [1/N for i = 1:N]
    return _compute_q(𝐏, 𝐏ₑ, base = base) / _qmax(dx, dy, base = base)
end
