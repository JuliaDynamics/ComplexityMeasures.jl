# TODO: What's this file? It's incomplete and with rogue code...

struct MultiscaleEntropy end

"""
    multiscale_entropy(x::AbstractVector)

"""
function multiscale_entropy end

function coarse_grain(x::AbstractVector{T}, τ) where T
    N = length(x)
    ys = Vector{T}(undef, 0)

    for j = 1:floor(Int, N/τ)
        yⱼ = 0.0
        for i = (j-1)*τ+1:j*τ
            yⱼ += x[i]
        end
        yⱼ *= 1/τ
        push!(ys, yⱼ)
    end
    return ys
end

x = rand(3 * 10^4 )
y1 = coarse_grain(x, 1)
y2 = coarse_grain(x, 2)
y3 = coarse_grain(x, 3)
y4 = coarse_grain(x, 4)
y5 = coarse_grain(x, 5)
y6 = coarse_grain(x, 6)
y7 = coarse_grain(x, 20)
