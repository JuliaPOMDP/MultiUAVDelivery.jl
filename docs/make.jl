using MultiUAVDelivery
using Documenter

makedocs(;
    modules=[MultiUAVDelivery],
    authors="rejuvyesh <mail@rejuvyesh.com> and contributors",
    repo="https://github.com/rejuvyesh/MultiUAVDelivery.jl/blob/{commit}{path}#L{line}",
    sitename="MultiUAVDelivery.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://rejuvyesh.github.io/MultiUAVDelivery.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rejuvyesh/MultiUAVDelivery.jl",
)
