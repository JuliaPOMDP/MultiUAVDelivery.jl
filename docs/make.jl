using MultiUAVDelivery
using Documenter

makedocs(;
    sitename="MultiUAVDelivery.jl",
    authors="Shushman <shushmanchoudhary@gmail.com>, rejuvyesh <mail@rejuvyesh.com> and contributors",
    modules=[MultiUAVDelivery],
    format=Documenter.HTML()
)

deploydocs(;
    repo="github.com/JuliaPOMDP/MultiUAVDelivery.jl",
)
