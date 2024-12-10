var documenterSearchIndex = {"docs":
[{"location":"SymbolicPermutation/#Permutation-(symbolic)","page":"Permutation (symbolic)","title":"Permutation (symbolic)","text":"","category":"section"},{"location":"SymbolicPermutation/","page":"Permutation (symbolic)","title":"Permutation (symbolic)","text":"SymbolicPermutation","category":"page"},{"location":"SymbolicPermutation/#Entropies.SymbolicPermutation","page":"Permutation (symbolic)","title":"Entropies.SymbolicPermutation","text":"SymbolicPermutation(m::Int; b::Real = 2)\n\nA symbolic permutation probabilities estimator using motifs of length m, based on Bandt & Pompe (2002)[BandtPompe2002].\n\nIf the estimator is used for entropy computation, then the entropy is computed  to base b (the default b = 2 gives the entropy in bits).\n\nThe motif length must be ≥ 2. By default m = 2, which is the shortest  possible permutation length which retains any meaningful dynamical information.\n\n[BandtPompe2002]: Bandt, Christoph, and Bernd Pompe. \"Permutation entropy: a natural complexity measure for time series.\" Physical review letters 88.17 (2002): 174102.\n\n\n\n\n\n","category":"type"},{"location":"SymbolicPermutation/","page":"Permutation (symbolic)","title":"Permutation (symbolic)","text":"entropy(x::Dataset{N, T}, est::SymbolicPermutation, α::Real = 1) where {N, T}","category":"page"},{"location":"SymbolicPermutation/#Entropies.entropy-Union{Tuple{T}, Tuple{N}, Tuple{Dataset{N,T},SymbolicPermutation}, Tuple{Dataset{N,T},SymbolicPermutation,Real}} where T where N","page":"Permutation (symbolic)","title":"Entropies.entropy","text":"entropy(x::Dataset, est::SymbolicPermutation, α::Real = 1)\nentropy!(s::Vector{Int}, x::Dataset, est::SymbolicPermutation, α::Real = 1)\n\nCompute the generalized order α permutation entropy of x, using symbol size est.m.\n\nProbability estimation\n\nAn unordered symbol frequency histogram is obtained by symbolizing the points in x, using probabilities(::Dataset{N, T}, ::SymbolicPermutation). Sum-normalizing this histogram yields a probability distribution over the symbols \n\nA pre-allocated symbol array s, where length(x) = length(s), can be provided to  save some memory allocations if the permutation entropy is to be computed for multiple data sets.\n\nEntropy estimation\n\nAfter the symbolization histogram/distribution has been obtained, the order α generalized entropy  is computed from that sum-normalized symbol distribution.\n\nNote: Do not confuse the order of the generalized entropy (α) with the order m of the  permutation entropy (est.m, which controls the symbol size). Permutation entropy is usually  estimated with α = 1, but the implementation here allows the generalized entropy of any  dimension to be computed from the symbol frequency distribution.\n\nLet p be an array of probabilities (summing to 1). Then the Rényi entropy is\n\nH_alpha(p) = frac11-alpha log left(sum_i pi^alpharight)\n\nand generalizes other known entropies, like e.g. the information entropy (alpha = 1, see [Shannon1948]), the maximum entropy (alpha=0, also known as Hartley entropy), or the correlation entropy (alpha = 2, also known as collision entropy).\n\n[Rényi1960]: A. Rényi, Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability, pp 547 (1960)\n\n[Shannon1948]: C. E. Shannon, Bell Systems Technical Journal 27, pp 379 (1948)\n\n\n\n\n\n","category":"method"},{"location":"SymbolicPermutation/","page":"Permutation (symbolic)","title":"Permutation (symbolic)","text":"probabilities(x::Dataset{N, T}, est::SymbolicPermutation) where {N, T}","category":"page"},{"location":"SymbolicPermutation/#Entropies.probabilities-Union{Tuple{T}, Tuple{N}, Tuple{Dataset{N,T},SymbolicPermutation}} where T where N","page":"Permutation (symbolic)","title":"Entropies.probabilities","text":"probabilities(x::Dataset, est::SymbolicPermutation)\nprobabilities!(s::Vector{Int}, x::Dataset, est::SymbolicPermutation)\n\nCompute the unordered probabilities of the occurrence of symbol sequences constructed from the data x.  A pre-allocated symbol array s, where length(x) = length(s), can be provided to  save some memory allocations if the probabilities are to be computed for multiple data sets.\n\n\n\n\n\n","category":"method"},{"location":"SymbolicPermutation/#Example","page":"Permutation (symbolic)","title":"Example","text":"","category":"section"},{"location":"SymbolicPermutation/","page":"Permutation (symbolic)","title":"Permutation (symbolic)","text":"This example reproduces the permutation entropy example on the logistic map from Bandt and Pompe (2002).","category":"page"},{"location":"SymbolicPermutation/","page":"Permutation (symbolic)","title":"Permutation (symbolic)","text":"using DynamicalSystems, PyPlot, Entropies\n\nds = Systems.logistic()\nrs = 3.5:0.001:4\nN_lyap, N_ent = 100000, 10000\n\n# Generate one time series for each value of the logistic parameter r\nlyaps, hs_entropies, hs_chaostools = Float64[], Float64[], Float64[]\n\nfor r in rs\n    ds.p[1] = r\n    push!(lyaps, lyapunov(ds, N_lyap))\n    \n    # For 1D systems `trajectory` returns a vector, so embed it using τs\n    # to get the correct 6d dimension on the embedding\n    x = trajectory(ds, N_ent)\n    τs = ([-i for i in 0:6-1]...,) # embedding lags\n    emb = genembed(x, τs)\n    \n    # Pre-allocate symbol vector, one symbol for each point in the embedding - this is faster!\n    s = zeros(Int, length(emb));\n    push!(hs_entropies, entropy!(s, emb, SymbolicPermutation(6, b = Base.MathConstants.e)))\n\n    # Old ChaosTools.jl style estimation\n    push!(hs_chaostools, permentropy(x, 6))\nend\n\nf = figure(figsize = (10,6))\na1 = subplot(311)\nplot(rs, lyaps); ylim(-2, log(2)); ylabel(\"\\$\\\\lambda\\$\")\na1.axes.get_xaxis().set_ticklabels([])\nxlim(rs[1], rs[end]);\n\na2 = subplot(312)\nplot(rs, hs_chaostools; color = \"C1\"); xlim(rs[1], rs[end]); \nxlabel(\"\\$r\\$\"); ylabel(\"\\$h_6 (ChaosTools.jl)\\$\")\n\na3 = subplot(313)\nplot(rs, hs_entropies; color = \"C2\"); xlim(rs[1], rs[end]); \nxlabel(\"\\$r\\$\"); ylabel(\"\\$h_6 (Entropies.jl)\\$\")\ntight_layout()\nsavefig(\"permentropy.png\")","category":"page"},{"location":"SymbolicPermutation/","page":"Permutation (symbolic)","title":"Permutation (symbolic)","text":"(Image: )","category":"page"},{"location":"SymbolicPermutation/#Utils","page":"Permutation (symbolic)","title":"Utils","text":"","category":"section"},{"location":"SymbolicPermutation/","page":"Permutation (symbolic)","title":"Permutation (symbolic)","text":"Some convenience functions for symbolization are provided.","category":"page"},{"location":"SymbolicPermutation/","page":"Permutation (symbolic)","title":"Permutation (symbolic)","text":"symbolize\nencode_motif","category":"page"},{"location":"SymbolicPermutation/#Entropies.symbolize","page":"Permutation (symbolic)","title":"Entropies.symbolize","text":"symbolize(x::Dataset{N, T}, est::SymbolicPermutation) where {N, T} → Vector{Int}\n\nSymbolize the vectors in x using Algorithm 1 from Berger et al. (2019)[Berger2019].\n\nThe symbol length is automatically determined from the dimension of the input data.\n\nExample\n\nComputing the order 5 permutation entropy for a 7-dimensional dataset.\n\nusing DelayEmbeddings, Entropies\nD = Dataset([rand(7) for i = 1:1000])\nsymbolize(D, SymbolicPermutation(5))\n\n[Berger2019]: Berger, Sebastian, et al. \"Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code.\" Entropy 21.10 (2019): 1023.\n\n\n\n\n\n","category":"function"},{"location":"SymbolicPermutation/#Entropies.encode_motif","page":"Permutation (symbolic)","title":"Entropies.encode_motif","text":"encode_motif(x, m::Int = length(x))\n\nEncode the length-m motif x (a vector of indices that would sort some vector v in ascending order)  into its unique integer symbol, using Algorithm 1 in Berger et al. (2019)[Berger2019].\n\nNote: no error checking is done to see if length(x) == m, so be sure to provide the correct motif length!\n\nExample\n\n# Some random vector\nv = rand(5)\n\n# The indices that would sort `v` in ascending order. This is now a permutation \n# of the index permutation (1, 2, ..., 5)\nx = sortperm(v)\n\n# Encode this permutation as an integer.\nencode_motif(x)\n\n[Berger2019]: Berger, Sebastian, et al. \"Teaching Ordinal Patterns to a Computer: Efficient Encoding Algorithms Based on the Lehmer Code.\" Entropy 21.10 (2019): 1023.\n\n\n\n\n\n","category":"function"},{"location":"histogram_estimation/","page":"-","title":"-","text":"non0hist","category":"page"},{"location":"histogram_estimation/","page":"-","title":"-","text":"binhist","category":"page"},{"location":"VisitationFrequency/#Visitation-frequency","page":"Visitation frequency","title":"Visitation frequency","text":"","category":"section"},{"location":"VisitationFrequency/","page":"Visitation frequency","title":"Visitation frequency","text":"VisitationFrequency","category":"page"},{"location":"VisitationFrequency/#Entropies.VisitationFrequency","page":"Visitation frequency","title":"Entropies.VisitationFrequency","text":"VisitationFreqency(r::RectangularBinning; b::Real = 2)\n\nA probability estimator based on binning data into rectangular boxes dictated by  the binning scheme r.\n\nIf the estimator is used for entropy computation, then the entropy is computed  to base b (the default b = 2 gives the entropy in bits).\n\nSee also: RectangularBinning.\n\n\n\n\n\n","category":"type"},{"location":"VisitationFrequency/","page":"Visitation frequency","title":"Visitation frequency","text":"entropy(x::Dataset, est::VisitationFrequency)","category":"page"},{"location":"VisitationFrequency/#Entropies.entropy-Tuple{Dataset,VisitationFrequency}","page":"Visitation frequency","title":"Entropies.entropy","text":"entropy(x::Dataset, est::VisitationFrequency, α::Real = 1) → Real\n\nEstimate the generalized order α entropy of x using a visitation frequency approach.  This is done by first estimating the sum-normalized unordered 1D histogram using probabilities, then computing entropy over that histogram/distribution.\n\nThe base b of the logarithms is inferred from the provided estimator  (e.g. est = VisitationFrequency(RectangularBinning(45), b = Base.MathConstants.e).\n\nDescription\n\nLet p be an array of probabilities (summing to 1). Then the Rényi entropy is\n\nH_alpha(p) = frac11-alpha log left(sum_i pi^alpharight)\n\nand generalizes other known entropies, like e.g. the information entropy (alpha = 1, see [Shannon1948]), the maximum entropy (alpha=0, also known as Hartley entropy), or the correlation entropy (alpha = 2, also known as collision entropy).\n\n[Rényi1960]: A. Rényi, Proceedings of the fourth Berkeley Symposium on Mathematics, Statistics and Probability, pp 547 (1960)\n\n[Shannon1948]: C. E. Shannon, Bell Systems Technical Journal 27, pp 379 (1948)\n\nSee also: VisitationFrequency, RectangularBinning.\n\nExample\n\nusing Entropies, DelayEmbeddings\nD = Dataset(rand(100, 3))\n\n# How shall the data be partitioned? Here, we subdivide each \n# coordinate axis into 4 equal pieces over the range of the data, \n# resulting in rectangular boxes/bins (see RectangularBinning).\nϵ = RectangularBinning(4)\n\n# Estimate entropy\nentropy(D, VisitationFrequency(ϵ))\n\n\n\n\n\n","category":"method"},{"location":"VisitationFrequency/","page":"Visitation frequency","title":"Visitation frequency","text":"probabilities(x::Dataset, est::VisitationFrequency)","category":"page"},{"location":"VisitationFrequency/#Entropies.probabilities-Tuple{Dataset,VisitationFrequency}","page":"Visitation frequency","title":"Entropies.probabilities","text":"probabilities(x::Dataset, est::VisitationFrequency) → Vector{Real}\n\nSuperimpose a rectangular grid (bins/boxes) dictated by est over the data x and return  the sum-normalized histogram (i.e. frequency at which the points of x visits the bins/boxes  in the grid) in an unordered 1D form, discarding all non-visited bins and bin edge information.\n\nPerformances Notes\n\nThis method has a linearithmic time complexity (n log(n) for n = length(data)) and a  linear space complexity l for l = dimension(data)). This allows computation of  histograms of high-dimensional datasets and with small box sizes ε without memory  overflow and with maximum performance.\n\nSee also: VisitationFrequency, RectangularBinning.\n\nExample\n\nusing Entropies, DelayEmbeddings\nD = Dataset(rand(100, 3))\n\n# How shall the data be partitioned? \n# Here, we subdivide each coordinate axis into 4 equal pieces\n# over the range of the data, resulting in rectangular boxes/bins\nϵ = RectangularBinning(4)\n\n# Feed partitioning instructions to estimator.\nest = VisitationFrequency(ϵ)\n\n# Estimate a probability distribution over the partition\nprobabilities(D, est)\n\n\n\n\n\n","category":"method"},{"location":"VisitationFrequency/#Specifying-binning/boxes","page":"Visitation frequency","title":"Specifying binning/boxes","text":"","category":"section"},{"location":"VisitationFrequency/","page":"Visitation frequency","title":"Visitation frequency","text":"RectangularBinning","category":"page"},{"location":"VisitationFrequency/#Entropies.RectangularBinning","page":"Visitation frequency","title":"Entropies.RectangularBinning","text":"RectangularBinning(ϵ) <: RectangularBinningScheme\n\nInstructions for creating a rectangular box partition using the binning scheme ϵ.  Binning instructions are deduced from the type of ϵ.\n\nRectangular binnings may be automatically adjusted to the data in which the RectangularBinning  is applied, as follows:\n\nϵ::Int divides each coordinate axis into ϵ equal-length intervals,   extending the upper bound 1/100th of a bin size to ensure all points are covered.\nϵ::Float64 divides each coordinate axis into intervals of fixed size ϵ, starting   from the axis minima until the data is completely covered by boxes.\nϵ::Vector{Int} divides the i-th coordinate axis into ϵ[i] equal-length   intervals, extending the upper bound 1/100th of a bin size to ensure all points are   covered.\nϵ::Vector{Float64} divides the i-th coordinate axis into intervals of fixed size ϵ[i], starting   from the axis minima until the data is completely covered by boxes.\n\nRectangular binnings may also be specified on arbitrary min-max ranges. \n\nϵ::Tuple{Vector{Tuple{Float64,Float64}},Int64} creates intervals   along each coordinate axis from ranges indicated by a vector of (min, max) tuples, then divides   each coordinate axis into an integer number of equal-length intervals. Note: this does not ensure   that all points are covered by the data (points outside the binning are ignored).\n\nExample 1: Grid deduced automatically from data (partition guaranteed to cover data points)\n\nFlexible box sizes\n\nThe following binning specification finds the minima/maxima along each coordinate axis, then  split each of those data ranges (with some tiny padding on the edges) into 10 equal-length  intervals. This gives (hyper-)rectangular boxes, and works for data of any dimension.\n\nusing Entropies\nRectangularBinning(10)\n\nNow, assume the data consists of 2-dimensional points, and that we want a finer grid along one of the dimensions than over the other dimension.\n\nThe following binning specification finds the minima/maxima along each coordinate axis, then  splits the range along the first coordinate axis (with some tiny padding on the edges)  into 10 equal-length intervals, and the range along the second coordinate axis (with some  tiny padding on the edges) into 5 equal-length intervals. This gives (hyper-)rectangular boxes.\n\nusing Entropies\nRectangularBinning([10, 5])\n\nFixed box sizes\n\nThe following binning specification finds the minima/maxima along each coordinate axis,  then split the axis ranges into equal-length intervals of fixed size 0.5 until the all data  points are covered by boxes. This approach yields (hyper-)cubic boxes, and works for  data of any dimension.\n\nusing Entropies\nRectangularBinning(0.5)\n\nAgain, assume the data consists of 2-dimensional points, and that we want a finer grid along one of the dimensions than over the other dimension.\n\nThe following binning specification finds the minima/maxima along each coordinate axis, then splits the range along the first coordinate axis into equal-length intervals of size 0.3, and the range along the second axis into equal-length intervals of size 0.1 (in both cases,  making sure the data are completely covered by the boxes). This approach gives a (hyper-)rectangular boxes. \n\nusing Entropies\nRectangularBinning([0.3, 0.1])\n\nExample 2: Custom grids (partition not guaranteed to cover data points):\n\nAssume the data consists of 3-dimensional points (x, y, z), and that we want a grid  that is fixed over the intervals [x₁, x₂] for the first dimension, over [y₁, y₂] for the second dimension, and over [z₁, z₂] for the third dimension. We when want to split each of those ranges into 4 equal-length pieces. Beware: some points may fall  outside the partition if the intervals are not chosen properly (these points are  simply discarded). \n\nThe following binning specification produces the desired (hyper-)rectangular boxes. \n\nusing Entropies, DelayEmbeddings\n\nD = Dataset(rand(100, 3));\n\nx₁, x₂ = 0.5, 1 # not completely covering the data, which are on [0, 1]\ny₁, y₂ = -2, 1.5 # covering the data, which are on [0, 1]\nz₁, z₂ = 0, 0.5 # not completely covering the data, which are on [0, 1]\n\nϵ = [(x₁, x₂), (y₁, y₂), (z₁, z₂)], 4 # [interval 1, interval 2, ...], n_subdivisions\n\nRectangularBinning(ϵ)\n\n\n\n\n\n","category":"type"},{"location":"VisitationFrequency/#Utils","page":"Visitation frequency","title":"Utils","text":"","category":"section"},{"location":"VisitationFrequency/","page":"Visitation frequency","title":"Visitation frequency","text":"Some convenience functions for symbolization are provided.","category":"page"},{"location":"VisitationFrequency/","page":"Visitation frequency","title":"Visitation frequency","text":"encode_as_bin\njoint_visits\nmarginal_visits","category":"page"},{"location":"VisitationFrequency/#Entropies.encode_as_bin","page":"Visitation frequency","title":"Entropies.encode_as_bin","text":"encode_as_bin(point, reference_point, edgelengths) → Vector{Int}\n\nEncode a point into its integer bin labels relative to some reference_point (always counting from lowest to highest magnitudes), given a set of box  edgelengths (one for each axis). The first bin on the positive side of  the reference point is indexed with 0, and the first bin on the negative  side of the reference point is indexed with -1.\n\nSee also: joint_visits, marginal_visits.\n\nExample\n\nusing Entropies\n\nrefpoint = [0, 0, 0]\nsteps = [0.2, 0.2, 0.3]\nencode_as_bin(rand(3), refpoint, steps)\n\n\n\n\n\n","category":"function"},{"location":"VisitationFrequency/#Entropies.joint_visits","page":"Visitation frequency","title":"Entropies.joint_visits","text":"joint_visits(points, binning_scheme::RectangularBinning) → Vector{Vector{Int}}\n\nDetermine which bins are visited by points given the rectangular binning scheme ϵ. Bins are referenced relative to the axis minima, and are  encoded as integers, such that each box in the binning is assigned a unique integer array (one element for each dimension). \n\nFor example, if a bin is visited three times, then the corresponding  integer array will appear three times in the array returned.\n\nSee also: marginal_visits, encode_as_bin.\n\nExample\n\nusing DelayEmbeddings, Entropies\n\npts = Dataset([rand(5) for i = 1:100]);\njoint_visits(pts, RectangularBinning(0.2))\n\n\n\n\n\n","category":"function"},{"location":"VisitationFrequency/#Entropies.marginal_visits","page":"Visitation frequency","title":"Entropies.marginal_visits","text":"marginal_visits(points, binning_scheme::RectangularBinning, dims) → Vector{Vector{Int}}\n\nDetermine which bins are visited by points given the rectangular binning scheme ϵ, but only along the desired dimensions dims. Bins are referenced  relative to the axis minima, and are encoded as integers, such that each box  in the binning is assigned a unique integer array (one element for each  dimension in dims). \n\nFor example, if a bin is visited three times, then the corresponding  integer array will appear three times in the array returned.\n\nSee also: joint_visits, encode_as_bin.\n\nExample\n\nusing DelayEmbeddings, Entropies\npts = Dataset([rand(5) for i = 1:100]);\n\n# Marginal visits along dimension 3 and 5\nmarginal_visits(pts, RectangularBinning(0.3), [3, 5])\n\n# Marginal visits along dimension 2 through 5\nmarginal_visits(pts, RectangularBinning(0.3), 2:5)\n\n\n\n\n\nmarginal_visits(joint_visits, dims) → Vector{Vector{Int}}\n\nIf joint visits have been precomputed using joint_visits, marginal  visits can be returned directly without providing the binning again  using the marginal_visits(joint_visits, dims) signature.\n\nSee also: joint_visits, encode_as_bin.\n\nExample\n\nusing DelayEmbeddings, Entropies\npts = Dataset([rand(5) for i = 1:100]);\n\n# First compute joint visits, then marginal visits along dimensions 1 and 4\njv = joint_visits(pts, RectangularBinning(0.2))\nmarginal_visits(jv, [1, 4])\n\n# Marginals along dimension 2\nmarginal_visits(jv, 2)\n\n\n\n\n\n","category":"function"},{"location":"#Entropies.jl","page":"Documentation","title":"Entropies.jl","text":"","category":"section"},{"location":"","page":"Documentation","title":"Documentation","text":"This package provides entropy estimators used for entropy computations in the CausalityTools.jl and DynamicalSystems.jl packages.","category":"page"},{"location":"","page":"Documentation","title":"Documentation","text":"Most of the code in this package assumes that your data is represented by the Dataset-type from DelayEmbeddings.jl, where each observation is a D-dimensional data point represented by a static vector. See the DynamicalSystems.jl documentation for more info.","category":"page"}]
}
