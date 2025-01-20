using Gen
using Distributions

@gen function beta_bernoulli_model(α::Float64, β::Float64, N::Int)
    # θ ~ Beta(α, β)
    θ = @trace(beta(α, β), :theta)
    # Data points Dᵢ ~ Bernoulli(θ)
    for i in 1:N
        @trace(bernoulli(θ), (:data, i))
    end
end


"""
    sample_beta_bernoulli_conjugate(model, α, β, data)

Given a model (the `@gen` function), hyperparameters α, β,
and the observed Bernoulli data (Vector{Bool} or Vector{Int}),
return a Gen trace where :theta is sampled *directly*
from the conjugate Beta posterior.
"""
function sample_beta_bernoulli_conjugate(model, α, β, data)
    # data can be Vector{Bool} or {0, 1}
    N = length(data)
    num_successes = sum(data)  # sum of 1's (true)
    α_post = α + num_successes
    β_post = β + (N - num_successes)

    # Sample θ ~ Beta(α_post, β_post)
    θ_draw = rand(Beta(α_post, β_post))

    # Construct the Gen trace manually
    tr = Gen.empty_trace(model)
    
    # Insert θ into the trace
    Gen.insert!.(tr, (:theta,), θ_draw)
    
    # Insert the observed data into the trace as well
    # so that the trace is consistent with the model specification.
    # We do not re-sample data from the model; we just
    # mark them as observed (since these are given).
    for i in 1:N
        Gen.insert!.(tr, (:data, i), data[i])
    end
    
    return tr
end

@gen function normal_invgamma_model(μ0::Float64, λ::Float64, α::Float64, β::Float64, N::Int)
    # sigma^2 ~ InvGamma(α, β)
    sigma2 = @trace(inverse_gamma(α, β), :sigma2)
    # μ ~ Normal(μ0, sqrt(sigma^2 / λ))
    μ     = @trace(normal(μ0, sqrt(sigma2 / λ)), :mu)
    
    # N data points
    for i in 1:N
        @trace(normal(μ, sqrt(sigma2)), (:x, i))
    end
end

"""
    gibbs_normal_invgamma(model, μ0, λ, α, β, data; num_iters=1000)

Run a Gibbs sampler for the Normal-Inverse-Gamma model defined in Gen.
Returns a vector of traces or final trace.
"""
function gibbs_normal_invgamma(
    model,
    μ0::Float64,
    λ::Float64,
    α::Float64,
    β::Float64,
    data::Vector{Float64};
    num_iters::Int=1000
)
    N = length(data)
    # Initialize (μ, sigma^2) - could do something more sophisticated
    mu_curr = mean(data)
    sigma2_curr = var(data) > 0 ? var(data) : 1.0  # fallback
    
    # We'll store intermediate samples if desired
    traces = Vector{Gen.Trace}()

    for iter in 1:num_iters
        # --- 1) Sample μ given σ^2, data
        x_bar = mean(data)
        mu_post_mean = (λ*μ0 + N*x_bar) / (λ + N)
        mu_post_var  = sigma2_curr / (λ + N)
        mu_curr      = rand(Normal(mu_post_mean, sqrt(mu_post_var)))

        # --- 2) Sample σ^2 given μ, data
        alpha_post = α + N/2
        # sum of squared deviations
        ssd = sum((x - mu_curr)^2 for x in data)
        beta_post = β + 0.5 * ssd
        sigma2_curr = rand(InverseGamma(alpha_post, beta_post))

        # Build a Gen trace for the current sample
        tr = Gen.empty_trace(model)
        # Insert latent variables
        Gen.insert!.(tr, (:sigma2,), sigma2_curr)
        Gen.insert!.(tr, (:mu,), mu_curr)
        # Insert observed data
        for i in 1:N
            Gen.insert!.(tr, (:x, i), data[i])
        end
        push!(traces, tr)
    end

    return traces
end