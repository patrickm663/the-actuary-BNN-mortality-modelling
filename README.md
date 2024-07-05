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

## References
[1] Bravo, J.M. 2021, Forecasting mortality rates with Recurrent Neural Networks: A preliminary investigation using Portuguese data. CAPSI 2021 Proceedings. 7. https://aisel.aisnet.org/capsi2021/7 
[2] Bennett, T.D., Russell, S., and Albers, D.J. 2021 Neural Networks for Mortality Prediction: Ready for Prime Time? Pediatr Crit Care Med. 2021 Jun 1;22(6):578-581. doi: 10.1097/PCC.0000000000002710. PMID: 34078844; PMCID: PMC8188609.
[3] Schnürch, S. and Korn, R. 2022. Point and interval forecasts of death rates using neural networks, ASTIN Bulletin, 52(1):333–360.
[4] Neal, R.M. 1994. Bayesian Learning for Neural Networks. Ph.D. thesis, University of Toronto, Canada.
[5] World Health Organization. 2024. Life tables: Life tables by country South Africa. Available at: https://apps.who.int/gho/data/view.searo.61540?lang=en
[6] Bishops, C.M. 1995. Neural networks for pattern recognition, chapter 10: Bayesian Techniques.  Department of Computer Science and Applied Mathematics, Aston University, Birmingham, UK.
[7] Hoffman, M.D. and Gelman, A. 2011. The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. arXiv preprint arXiv:1111.4246.
[8] Neal, R. 1994. An improved acceptance procedure for the hybrid Monte Carlo algorithm. Journal of Computational Physics, 111:194–203.
[9] Neal, R. 2011. Handbook of Markov Chain Monte Carlo. CRC Press.
[10] Graves, A. 2011. Practical variational inference for neural networks. Department of Computer Science, University of Toronto, Canada
[11] Lee, R.D. and Carter, L.R. 1992. Modeling and Forecasting U.S. Mortality, Journal of the American Statistical Association, 87:419, 659-671
[12] Ramachandran, P., Zoph, B., and Le, Q.V. 2017. Swish: a self-gated activation function, Google Brain. arXiv:1710.05941
[13] Avik, P. 2023. Lux: Explicit Parameterization of Deep Neural Networks in Julia (v0.5.41). Zenodo, Available at:  https://github.com/LuxDL/Lux.jl
[14] Ge, H., Xu, K., and Ghahramani, Z. 2018. Turing: a language for flexible probabilistic inference, International Conference on Artificial Intelligence and Statistics, AISTATS 2018, 1682-1690. Available at: https://github.com/TuringLang/Turing.jl
[15] Innes, M. 2019. Tracker.jl. Available at: https://github.com/FluxML/Tracker.jl
[16] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and Bürkner, P. 2019. Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. arXiv preprint arXiv:1903.08008
[17] Cloudenback, A. 2024. JuliaActuary. Available at: https://juliaactuary.org

## License
MIT licensed.

## TODO
- [ ] Clean-up notebook
- [ ] Link to article once available
- [ ] Add summary of results in README


