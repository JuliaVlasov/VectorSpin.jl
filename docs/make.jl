using VectorSpin
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(VectorSpin, :DocTestSetup, :(using VectorSpin); recursive = true)


bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:authoryear
)

makedocs(;
    modules = [VectorSpin],
    authors = "Julia Vlasov",
    sitename = "VectorSpin.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaVlasov.github.io/VectorSpin.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Hamiltonian Splitting" => "hamiltonian_splitting.md",
        "Numerical scheme" => "numerical_scheme.md",
        "Validation" => "srs_without_spin.md",
        "Vlasov-Maxwell" => "example.md",
        "Version with ions" => "spin_ions.md",
        "API" => "api.md",
    ],
    plugins=[bib]
)

deploydocs(; repo = "github.com/JuliaVlasov/VectorSpin.jl", devbranch = "main")
