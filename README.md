# BNN Mortality Modelling
Some applications of Bayesian Neural Networks for submission to The Actuary Magazine.

## Set-up
This notebook can be run as follows:
1. Ensure Julia is installed
2. Clone the repo and enter the directory
3. Start the Julia REPL and instantiate the project (i.e. install the dependencies)
```
julia --project=.
```
```julia
julia> ] ## to enter Pkg mode
(the-actuary-BNN-mortality-m...) pkg> instantiate ## installs the dependencies
```
4. Exit Pkg mode using `Ctrl+C`
5. While still in the REPL, launch Pluto
```julia
julia> using Pluto
julia> Pluto.run()
```
6. Navigate to http://localhost:1234 in your browser (if not automatically redirected)
7. Open `notebook/main.jl` from Pluto's user interface

## Results
Graphs can be viewed in `results/`. HTML output of the notebook can be viewed in `results/main.html`.

## License
MIT licensed.
