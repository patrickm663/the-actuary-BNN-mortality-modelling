### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ b6d1ee8e-e047-11ee-0264-a393c35d00de
begin
	using Pkg
	cd("../")
	Pkg.activate(".")
end

# ╔═╡ 89ce0d44-2db7-4232-afec-59d6453b3348
using Turing, StatsPlots, Lux, Tracker, LinearAlgebra, Random, DataFrames, Functors, Distributions, CSV

# ╔═╡ 56913eb2-b189-4cf0-8019-62da29df89ab
begin
	rng = Random.default_rng()
	Random.seed!(rng, 456789)
end

# ╔═╡ ec122fd6-ae50-4da3-8345-8232a59d2bae
begin
	df = DataFrame(CSV.File("data/sa_who_data.csv"))[2:end, 1:end]
	df = df[:, ["2019", "2015", "2010", "2005", "2000"]]
	rename!(df, ["2019", "2015", "2010", "2005", "2000"])
	df = parse.(Float64, df)
end

# ╔═╡ 0046858a-1c64-4303-b383-b2202e954d89
begin
	X = [2000 2005 2010 2015 2019]'
	#y = Matrix(df[[6, 8, 11, 15, 18], :])'
	y = Matrix(df[6:end, :])
	
	#age_range = ["20-24", "30-34", "45-49", "65-69", "80-84"]
	#age_range_int = [22.5, 32.5, 47.5, 67.5, 83.5]
	age_range = ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85+"]
	age_range_int = [22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 83.5, 87.5]
end

# ╔═╡ 2fa42fc5-463a-4644-89ba-97fa0cb63381
begin
	plot(xlab="Age", ylab="log(μ)", title="South African Mortality:\n All Lives", legend=:outerright, xlim=(20,90))
	for yrs ∈ 1:length(X)
		scatter!(age_range_int, log.(y)[:, yrs], label="$(X[yrs])", width=.8)
	end
	plot!(age_range_int, mean(log.(y); dims=2), label="Average", color=:Red, width=2)
end

# ╔═╡ bb40231f-b691-4886-bc4e-cc1fb81a7971
function lee_carter(d)
	log_A = log.(Matrix(d))
	αₓ = mean(log_A; dims=2)
	log_A_std = log_A .- αₓ
	U, λ, Vt = svd(log_A_std)
	bₓ = U[:, 1]
	kₜ = λ[1] .* Vt[:, 1]
	log_LC = αₓ .+ bₓ * kₜ'
	
	return log_LC, αₓ, bₓ, kₜ
end

# ╔═╡ 60745cc8-02ff-4b17-a531-0bee6d21a321
log_LC, αₓ, bₓ, kₜ = lee_carter(y)

# ╔═╡ 3280ec10-969e-4861-8229-07d2e79b4de0
sum(bₓ)

# ╔═╡ 3efb1657-bdba-4eb3-95bb-b2c049cbabad
sum(kₜ)

# ╔═╡ b5643ba7-5bc0-47f4-a221-3480fde60609
plot(X, kₜ, label="kₜ", width=2, xlab="Year", ylab="log(μₜ)")

# ╔═╡ 9ddf6186-75fd-477a-9e28-dde510589e07
plot(age_range_int, αₓ, label="αₓ", width=2, xlab="Age", ylab="log(μₓ)")

# ╔═╡ d190daec-454d-49b3-8a9f-3506aaa9e57b
plot(age_range_int, bₓ, label="bₓ", width=2, xlab="Age", ylab="log(μₓ)")

# ╔═╡ 8dedce0c-945c-4ec9-8613-040df7f6ca3b
function standardise(x)
	μ = mean(x; dims=1)
	σ = std(x; dims=1)
	stan = (x .- μ) ./ σ
	return (stan, μ, σ)
end

# ╔═╡ 9e52366d-5d72-4315-be9e-c79845180e45
rescale(x, μ, σ) = (x .* σ) .+ μ

# ╔═╡ 754c6cd9-d8c7-464f-8260-364abcfd77dc
begin
	Xs, μ_X, σ_X = standardise(X)
	ys, μ_y, σ_y = standardise(log.(y')) # y scaled using log rates
	log_LC_s, _, _ = standardise(log_LC')
end

# ╔═╡ 7af3f0e0-878e-402d-86f9-78630c00a89c
begin
	# Construct a neural network using Lux
	nn = Chain(
		Dense(size(Xs)[2] => 8, swish), 
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => 8, swish),
		Dense(8 => size(ys)[2]))

	# Initialize the model weights and state
	ps, st = Lux.setup(rng, nn)

	# Create a regularization term and a Gaussian prior variance term.
	alpha = 0.8
	sig = 1.0 / sqrt(alpha)
	@show sig, sig^2, alpha, Lux.parameterlength(nn)

	function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
	    @assert length(ps_new) == Lux.parameterlength(ps)
	    i = 1
	    function get_ps(x)
	        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
	        i += length(x)
	        return z
	    end
	    return fmap(get_ps, ps)
	end

	# Specify the probabilistic model.
	@model function bayes_nn(xs, y, ::Type{T} = Float64) where {T}
	    global st

		years, ages = size(y)

		# HalfNormal prior on σ
		σ = Vector{Float64}(undef, ages)
	
	    # Sample the parameters from a MvNormal with 0 mean and constant variance
	    nparameters = Lux.parameterlength(nn)
	    parameters ~ MvNormal(zeros(nparameters), (sig^2) .* I)
	
	    # Forward NN to make predictions
	    preds, st = nn(xs, vector_to_parameters(parameters, ps), st)
	
	    # Age-range log(μ) are each given a MvNormal likelihood with constant variance
	    for j in 1:ages
			σ[j] ~ truncated(Normal(0, 1); lower=1e-9) # Per age-band
	        y[:, j] ~ MvNormal(preds[j, :], (σ[j]^2) .* I)
	    end
		return Nothing
	end
end

# ╔═╡ 7968a394-a3fc-4ec7-871a-6d6272cf29af
begin
	N = 12_500
	half_N = Int(ceil(N/2))
	ch = sample(bayes_nn(Xs', ys), NUTS(0.9; adtype=AutoTracker()), N; discard_adapt=false)
end

# ╔═╡ 95371d4b-c937-4344-a69f-c9b0f7b12893
θ = MCMCChains.group(ch, :parameters).value;

# ╔═╡ 9f26a81b-326a-4938-a946-689b14f70d83
plot(ch[half_N:end, 1:80:end, :])

# ╔═╡ 83ce8ad4-2b61-48ef-94f1-b65080a813aa
describe(ch)

# ╔═╡ f4b94fb1-0a0c-456c-af46-2c7a47eb4f6c
nn_forward(x, θ) = first(nn(x, vector_to_parameters(θ, ps), st))

# ╔═╡ f10f309d-1a25-4e67-80f5-9c4dd8978e60
begin
	# Find the index that provided the highest log posterior in the chain.
	_, idx_ = findmax(ch[:lp])
	
	# Extract the max row value from i.
	idx_ = idx_.I[1]
end

# ╔═╡ fb42520b-b0b0-4692-bc2d-9327978d4566
begin
	Xs_full = 1990:0.1:2030
	Xs_fulls = (Xs_full .- μ_X) ./ σ_X # Use same as train/test standardisation
end;

# ╔═╡ e61bc60c-a0f1-4fd1-905c-7e7459df9bba
μ_y

# ╔═╡ d96f5bba-8785-49a7-b0c4-edae3c6b2cea
σ_y

# ╔═╡ 22f4183d-0518-4f8b-9656-645ee26ddf32
function age_plot(year)
	@assert year ≤ length(age_range)

	sample_N = 100_000
	nn_pred_samples = Matrix{Float64}(undef, sample_N, length(Xs_full))
	posterior_samples = sample(ch, sample_N)
	θ_samples = MCMCChains.group(posterior_samples, :parameters).value;

	# BNN samples
	for i ∈ 1:sample_N
		nn_pred_y = rescale(nn_forward(Xs_fulls', θ_samples[i, :])', μ_y, σ_y)[:, year]
		nn_pred_samples[i, :] .= nn_pred_y
	end

	nn_pred_mean = mean(nn_pred_samples; dims=1)'
	nn_pred_median = quantile.(eachcol(nn_pred_samples), 0.50)
	nn_pred_l25 = quantile.(eachcol(nn_pred_samples), 0.25)
	nn_pred_u75 = quantile.(eachcol(nn_pred_samples), 0.75)
	
	p_ = begin
		# Plot front-matter
		plot(title="South African Log-Mortality Rates\n $(age_range[year])", xlab="Year", ylab="log(μ)", label="", legend=:outertopright)
		
		plot!(Xs_full, nn_pred_l25, fillrange = nn_pred_u75, label="BNN: 25-75% CI", color=:blue, width=.1, alpha=.3)
	
		# Plot BNN mean
		plot!(Xs_full, nn_pred_mean, label="BNN: Mean", color=:blue, width=2)

		# Plot BNN median
		plot!(Xs_full, nn_pred_median, label="BNN: Median", color=:blue, width=2, style=:dashdot)

		# Plot Lee-Carter (all lives)
		plot!(X, log_LC[year, :], label="Lee-Carter", width=2, color=:brown, style=:dash)
	
		# Plot Observations
		scatter!(X, log.(y)[year, :], label="Observations", color=:black, markershape=:circle)
	end

	savefig(p_, "$(age_range[year])-BNN.svg")
	
	return p_
end

# ╔═╡ 8068a5ae-3e8e-4e2b-9099-b2a0494a9589
age_plot(1)

# ╔═╡ 9af612d8-0aed-40a1-8e58-292e2650a6bf
age_plot(2)

# ╔═╡ 5e87d7da-7d00-4189-9993-fd05166fde11
age_plot(3)

# ╔═╡ 2aaf5999-838c-4e65-a472-a7b091c49c83
age_plot(4)

# ╔═╡ ffb37f6b-15ff-43a8-828e-e4d93eaf68cc
age_plot(5)

# ╔═╡ 6926a959-a50f-460f-93ff-4e17473670c1
age_plot(6)

# ╔═╡ 7081f671-2079-43ee-978d-81795ec9c498
age_plot(7)

# ╔═╡ 835c5d4e-bfbe-4fcf-adcd-5537283f0abf
age_plot(8)

# ╔═╡ 73c6250e-9c3e-4fcf-81f0-8741d77169cf
age_plot(9)

# ╔═╡ ee6d5f3d-ebb8-4057-b3c4-0a6191e118c7
age_plot(10)

# ╔═╡ aadf5d17-4dfe-4e45-a0ee-07218c449fb3
age_plot(11)

# ╔═╡ d17b32c0-1404-477f-8dc8-6fce6a3d3ec5
age_plot(12)

# ╔═╡ b2628231-ef62-46b6-a6fe-09cae61dbc93
age_plot(13)

# ╔═╡ 21024292-2cdc-4bb6-b1cc-592e4816fe18
age_plot(14)

# ╔═╡ 8fe60e4f-ff41-4091-9f79-49b1512e4f05
print("MSE BNN MAP: ", mean((nn_forward(Xs', θ[idx_, :])' .- ys) .^ 2))

# ╔═╡ 43997904-6590-44d4-9415-709908bb5248
print("MSE LC: ", mean((log_LC_s .- ys) .^ 2))

# ╔═╡ 1503076a-bab9-45ff-bfc0-a2ccd8dd375d
print("MSE Ratio: ", mean((log_LC_s .- ys) .^ 2) / mean((nn_forward(Xs', θ[idx_, :])' .- ys) .^ 2))

# ╔═╡ 9ff05b2a-86f3-41b6-928a-6dc4359a983f
mean((nn_forward(Xs', θ[idx_, :])' .- ys) .^ 2; dims=1)'

# ╔═╡ c5bbfd64-1b7d-43fc-88e1-c2af73327a25
mean((log_LC_s .- ys) .^ 2; dims=1)'

# ╔═╡ Cell order:
# ╠═b6d1ee8e-e047-11ee-0264-a393c35d00de
# ╠═89ce0d44-2db7-4232-afec-59d6453b3348
# ╠═56913eb2-b189-4cf0-8019-62da29df89ab
# ╠═ec122fd6-ae50-4da3-8345-8232a59d2bae
# ╠═0046858a-1c64-4303-b383-b2202e954d89
# ╠═2fa42fc5-463a-4644-89ba-97fa0cb63381
# ╠═bb40231f-b691-4886-bc4e-cc1fb81a7971
# ╠═60745cc8-02ff-4b17-a531-0bee6d21a321
# ╠═3280ec10-969e-4861-8229-07d2e79b4de0
# ╠═3efb1657-bdba-4eb3-95bb-b2c049cbabad
# ╠═b5643ba7-5bc0-47f4-a221-3480fde60609
# ╠═9ddf6186-75fd-477a-9e28-dde510589e07
# ╠═d190daec-454d-49b3-8a9f-3506aaa9e57b
# ╠═8dedce0c-945c-4ec9-8613-040df7f6ca3b
# ╠═9e52366d-5d72-4315-be9e-c79845180e45
# ╠═754c6cd9-d8c7-464f-8260-364abcfd77dc
# ╠═7af3f0e0-878e-402d-86f9-78630c00a89c
# ╠═7968a394-a3fc-4ec7-871a-6d6272cf29af
# ╠═95371d4b-c937-4344-a69f-c9b0f7b12893
# ╠═9f26a81b-326a-4938-a946-689b14f70d83
# ╠═83ce8ad4-2b61-48ef-94f1-b65080a813aa
# ╠═f4b94fb1-0a0c-456c-af46-2c7a47eb4f6c
# ╠═f10f309d-1a25-4e67-80f5-9c4dd8978e60
# ╠═fb42520b-b0b0-4692-bc2d-9327978d4566
# ╠═e61bc60c-a0f1-4fd1-905c-7e7459df9bba
# ╠═d96f5bba-8785-49a7-b0c4-edae3c6b2cea
# ╠═22f4183d-0518-4f8b-9656-645ee26ddf32
# ╠═8068a5ae-3e8e-4e2b-9099-b2a0494a9589
# ╠═9af612d8-0aed-40a1-8e58-292e2650a6bf
# ╠═5e87d7da-7d00-4189-9993-fd05166fde11
# ╠═2aaf5999-838c-4e65-a472-a7b091c49c83
# ╠═ffb37f6b-15ff-43a8-828e-e4d93eaf68cc
# ╠═6926a959-a50f-460f-93ff-4e17473670c1
# ╠═7081f671-2079-43ee-978d-81795ec9c498
# ╠═835c5d4e-bfbe-4fcf-adcd-5537283f0abf
# ╠═73c6250e-9c3e-4fcf-81f0-8741d77169cf
# ╠═ee6d5f3d-ebb8-4057-b3c4-0a6191e118c7
# ╠═aadf5d17-4dfe-4e45-a0ee-07218c449fb3
# ╠═d17b32c0-1404-477f-8dc8-6fce6a3d3ec5
# ╠═b2628231-ef62-46b6-a6fe-09cae61dbc93
# ╠═21024292-2cdc-4bb6-b1cc-592e4816fe18
# ╠═8fe60e4f-ff41-4091-9f79-49b1512e4f05
# ╠═43997904-6590-44d4-9415-709908bb5248
# ╠═1503076a-bab9-45ff-bfc0-a2ccd8dd375d
# ╠═9ff05b2a-86f3-41b6-928a-6dc4359a983f
# ╠═c5bbfd64-1b7d-43fc-88e1-c2af73327a25
