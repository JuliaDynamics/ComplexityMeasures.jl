cd(@__DIR__)
using Pkg
CI = get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
CI && Pkg.activate(@__DIR__)
CI && Pkg.instantiate()
CI && (ENV["GKSwstype"] = "100")
using DelayEmbeddings
using Documenter
using DocumenterTools: Themes
using Entropies
using PyPlot
using DynamicalSystems
using Wavelets

# %% JuliaDynamics theme.
# download the themes
using DocumenterTools: Themes
for file in ("juliadynamics-lightdefs.scss", "juliadynamics-darkdefs.scss", "juliadynamics-style.scss")
    download("https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/$file", joinpath(@__DIR__, file))
end
# create the themes
for w in ("light", "dark")
    header = read(joinpath(@__DIR__, "juliadynamics-style.scss"), String)
    theme = read(joinpath(@__DIR__, "juliadynamics-$(w)defs.scss"), String)
    write(joinpath(@__DIR__, "juliadynamics-$(w).scss"), header*"\n"*theme)
end
# compile the themes
Themes.compile(joinpath(@__DIR__, "juliadynamics-light.scss"), joinpath(@__DIR__, "src/assets/themes/documenter-light.css"))
Themes.compile(joinpath(@__DIR__, "juliadynamics-dark.scss"), joinpath(@__DIR__, "src/assets/themes/documenter-dark.css"))

# %% Build docs
PyPlot.ioff()
cd(@__DIR__)
ENV["JULIA_DEBUG"] = "Documenter"

PAGES = [
    "Entropies.jl" => "index.md",
    "Estimators" => [
        "CountOccurrences.md",
        "VisitationFrequency.md",
        "TransferOperator.md",
        "Permutation.md",
        "NearestNeighbors.md",
        "NaiveKernel.md",
        "TimeScaleMODWT.md",
        "Walkthrough.md"
    ],
    "Non-exported" => "nonexported.md"
]

makedocs(
    modules = [Entropies],
    format = Documenter.HTML(
        prettyurls = CI,
        assets = [
            asset("https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap", class=:css),
        ],
        ),
    sitename = "Entropies.jl",
    authors = "Kristian Agasøster Haaga, George Datseris",
    pages = PAGES
)

if CI
    deploydocs(
        repo = "github.com/JuliaDynamics/Entropies.jl.git",
        target = "build",
        push_preview = true
    )
end
PyPlot.close("all")
PyPlot.ion()
