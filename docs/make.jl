cd(@__DIR__)
using ComplexityMeasures

# Convert tutorial file to markdown
import Literate
Literate.markdown("src/tutorial.jl", "src"; credit = false)
Literate.markdown("src/measure_count.jl", "src"; credit = false)

import Documenter

pages = [
    "index.md",
    "tutorial.md",
    "probabilities.md",
    "information_measures.md",
    "complexity.md",
    "convenience.md",
    "examples.md",
    "devdocs.md",
    "references.md",
    Documenter.hide("measure_count.md"),
]

# For easier debugging when downloading from a specific branch.
github_user = "JuliaDynamics"
branch = "master"
download_path = "https://raw.githubusercontent.com/$github_user/doctheme/$branch/"

import Downloads
Downloads.download(
    "$download_path/build_docs_with_style.jl",
    joinpath(@__DIR__, "build_docs_with_style.jl")
)
include("build_docs_with_style.jl")

using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "refs.bib");
    style=:authoryear
)

build_docs_with_style(pages, ComplexityMeasures, StateSpaceSets;
    expandfirst = ["index.md"], bib,
)
