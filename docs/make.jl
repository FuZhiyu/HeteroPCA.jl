using HeteroPCA
using Documenter

DocMeta.setdocmeta!(HeteroPCA, :DocTestSetup, :(using HeteroPCA); recursive=true)

makedocs(;
    modules=[HeteroPCA],
    authors="Zhiyu Fu <fuzhiyu0@gmail.com> and contributors",
    repo="https://github.com/fuzhiyu/HeteroPCA.jl/blob/{commit}{path}#{line}",
    sitename="HeteroPCA.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fuzhiyu.github.io/HeteroPCA.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fuzhiyu/HeteroPCA.jl",
)
