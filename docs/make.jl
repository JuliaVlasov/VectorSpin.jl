using VectorSpin
using Documenter

DocMeta.setdocmeta!(VectorSpin, :DocTestSetup, :(using VectorSpin); recursive=true)

makedocs(;
    modules=[VectorSpin],
    authors="Julia Vlasov",
    sitename="VectorSpin.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaVlasov.github.io/VectorSpin.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Validation" => "srs_without_spin.md",
        "Example" => "example.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaVlasov/VectorSpin.jl",
    devbranch="main",
)
