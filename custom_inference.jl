# using Gen, Distributions, SpecialFunctions

# """
#     to_z(value, dist) -> z, logdet_jac

# Given `value` drawn from `dist`, produce an equivalent `z ~ Normal(0,1)` 
# and the log of the absolute determinant of the Jacobian if needed.
# We store both so we can build a correct log-density in z-space.
# """
# function to_z(value::Float64, dist)
#     if dist isa Normal
#         # dist = Normal(μ, σ)
#         μ, σ = dist.μ, dist.σ
#         z = (value - μ) / σ
#         # The transform x -> (x-μ)/σ is linear, so log|Jac|= -log(σ).
#         logdet_jac = -log(σ)
#         return z, logdet_jac

#     elseif dist isa Beta
#         # dist = Beta(α, β)
#         α, β = dist.α, dist.β
#         # 1) u = cdf(Beta(α,β), value) in (0,1)
#         u = cdf(dist, value)
#         # 2) z = Φ⁻¹(u) in ℝ
#         z = quantile(Normal(0,1), u)
#         #
#         # logdet_jac = log( dθ/dz ) ? We’ll handle that by comparing
#         #  logpdf(Normal(0,1), z) - logpdf(Beta(α,β), θ)
#         #  in the final "prior_correction". So we can simply do:
#         logdet_jac = 0.0
#         return z, logdet_jac

#     elseif dist isa InverseGamma
#         # dist = InverseGamma(α, θ)
#         αig, βig = dist.α, dist.β
#         # For x ~ InvGamma(α,β), we can do the transform:
#         #   u = cdf(InvGamma(α,β), x) in (0,1)
#         #   z = Φ⁻¹(u).
#         # Same logic as Beta. (Or you might prefer log(x) -> standard normal, etc.)
#         u = cdf(dist, value)
#         z = quantile(Normal(), u)
#         logdet_jac = 0.0
#         return z, logdet_jac

#     elseif dist isa Gamma
#         # For x ~ Gamma(k, θ), or equivalently shape=k, rate=1/θ or something.
#         # We'll assume dist in Julia has shape α, rate β or shape-rate parameterization.
#         # We do same cdf -> Normal transform:
#         u = cdf(dist, value)
#         z = quantile(Normal(), u)
#         logdet_jac = 0.0
#         return z, logdet_jac

#     elseif dist isa Exponential
#         # Exponential(λ). Same idea:
#         u = cdf(dist, value)
#         z = quantile(Normal(), u)
#         logdet_jac = 0.0
#         return z, logdet_jac

#     else
#         # fallback: treat as if "value" is standard normal. Or do a random-walk update.
#         # For demonstration, we do a no-op "z = value", ignoring transforms.
#         z = value
#         logdet_jac = 0.0
#         return z, logdet_jac
#     end
# end

# """
#     from_z(z, dist) -> value

# Inverse transform: from z in ℝ to the domain of `dist`.
# """
# function from_z(z::Float64, dist)
#     if dist isa Normal
#         μ, σ = dist.μ, dist.σ
#         return μ + σ*z

#     elseif dist isa Beta
#         α, β = dist.α, dist.β
#         # convert z -> u via Φ(z), then u -> θ via invcdf(Beta(α,β), u)
#         u = cdf(Normal(), z)
#         # handle edge cases if u is 0 or 1 numerically
#         return quantile(Beta(α, β), u)

#     elseif dist isa InverseGamma
#         # same logic
#         u = cdf(Normal(), z)
#         return quantile(dist, u)

#     elseif dist isa Gamma
#         u = cdf(Normal(), z)
#         return quantile(dist, u)

#     elseif dist isa Exponential
#         u = cdf(Normal(), z)
#         return quantile(dist, u)

#     else
#         # fallback (no transform)
#         return z
#     end
# end


# """
#     elliptical_slice_1d(
#         trace, address, dist;
#         log_joint, num_slice_draws=1
#     )

# Performs 1D elliptical slice sampling updates on the random choice at `address`,
# given its prior distribution `dist`, within a Gen trace.
# `log_joint(trace)` is a function that returns log of the *current* model joint
# (including constraints). We do `num_slice_draws` repeated ESS updates.
# Returns nothing (operates in-place).
# """
# function elliptical_slice_1d(
#     trace,
#     address,
#     dist;
#     log_joint,
#     num_slice_draws::Int=1
# )
#     # -- 0) Convert the current value -> z in R
#     current_value = trace[address]
#     z0, _ = to_z(current_value, dist)

#     # We'll need the log of the "full posterior" in z-space. 
#     # But we treat the "prior" as N(0,1) in z-space, so we do a correction:
#     # 
#     #    log_posterior(z) = log_joint_of( from_z(z) ) 
#     #                       + logpdf(N(0,1), z) 
#     #                       - logpdf(dist, from_z(z))
#     # 
#     # Because logpdf(trace) includes logpdf(dist, current_value).
#     # We'll define:
#     function log_posterior_z(z::Float64)
#         # 1) Convert z -> value
#         val = from_z(z, dist)
#         # 2) Temporarily set trace
#         Gen.setval!(trace, address, val)
#         # 3) Evaluate the model's joint density
#         lj = log_joint(trace)
#         # 4) Add the "reparam prior correction"
#         correction = logpdf(Normal(0,1), z) - logpdf(dist, val)
#         return lj + correction
#     end

#     current_logpost = log_posterior_z(z0)

#     for _ in 1:num_slice_draws
#         # (a) random direction from N(0,1) in 1D is just a single real
#         nu = rand(Normal(0,1))

#         # (b) propose slice threshold
#         threshold = log(rand()) + current_logpost

#         # (c) bracket in [-π, +π]
#         theta = 2π * rand()
#         theta_min = theta - 2π
#         theta_max = theta

#         z_cur = z0  # current "z"

#         # (d) shrink bracket until acceptance
#         while true
#             z_prop = z_cur*cos(theta) + nu*sin(theta)
#             lp_prop = log_posterior_z(z_prop)
#             if lp_prop >= threshold
#                 # accept
#                 z_cur = z_prop
#                 current_logpost = lp_prop
#                 break
#             else
#                 # shrink bracket
#                 if theta < 0
#                     theta_min = theta
#                 else
#                     theta_max = theta
#                 end
#                 theta = rand()*(theta_max - theta_min) + theta_min
#             end
#         end

#         # done one slice iteration
#         z0 = z_cur
#     end
# end

# """
#     universal_elliptical_inference(
#         model, model_args;
#         constraints=choicemap(),
#         num_sweeps=1000
#     ) -> trace

# A single function that can handle *all* of your example models. 
# It does "blocked" updates address by address, using elliptical slice sampling
# if (a) the prior distribution is known and transformable to a standard Normal,
# or (b) fallback to a small random-walk MH otherwise.

# For simplicity, we do 1 ESS update per address per sweep.

# Returns a final DynamicTrace.
# """
# function universal_elliptical_inference(
#     model::Function,
#     model_args::Tuple;
#     constraints=Gen.choicemap(),
#     num_sweeps::Int=1000
# )
#     # 1) Generate an initial trace consistent with constraints
#     trace, _ = Gen.generate(model, model_args, constraints)

#     # 2) We'll define a function to compute the model's log joint:
#     log_joint(trace) = Gen.logpdf(trace)

#     # 3) Identify all addresses. We'll update them in a systematic order. 
#     #    get_choices(trace) returns an AddressValue pair for each random choice
#     addresses = collect(keys(get_choices(trace)))

#     # 4) For each sweep:
#     for s in 1:num_sweeps
#         # Shuffle or iterate in a fixed order. We'll do a fixed order here:
#         for addr in addresses
#             # get the distribution from the trace
#             dist = Gen.get_choice_distribution(trace, addr)
#             # We'll do elliptical slice if it's one of our recognized types
#             # or if there's a reparam transform in `to_z/from_z`.
#             known_dists = (Normal, Beta, InverseGamma, Gamma, Exponential)
#             if any(d -> dist isa d, known_dists)
#                 # 1D elliptical slice update
#                 elliptical_slice_1d(
#                     trace, addr, dist;
#                     log_joint=log_joint, num_slice_draws=1
#                 )
#             else
#                 # fallback to a trivial random-walk MH (not shown).
#                 # e.g., do_nuts_mh_update(trace, addr) or something:
#                 # We'll just skip for brevity.
#             end
#         end
#     end

#     return trace
# end

# # Example: log_poisson
# m0, t2 = 0.0, 2.0
# obs = choicemap((:y)=> 10)  # Suppose y=10 is observed
# final_trace = universal_elliptical_inference(log_poisson, (m0, t2); 
#                                              constraints=obs, 
#                                              num_sweeps=2000)
# println("Log pdf of final trace = ", Gen.logpdf(final_trace))
# println("Posterior sample for logtheta = ", final_trace[:logtheta])


# using Gen, Distributions, Plots, StatsPlots

# # A distribution that is guaranteed to be 1 or higher.
# @dist poisson_plus_one(rate) = poisson(rate) + 1;

# function logmeanexp(scores)
#     logsumexp(scores) - log(length(scores))
# end;

# # Define the piecewise_constant model
# @gen function piecewise_constant(xs::Vector{Float64})
#     # Generate a number of segments (at least 1)
#     segment_count ~ poisson_plus_one(1)
    
#     # Draw a vector on the simplex from a Dirichlet distribution
#     fractions ~ Gen.dirichlet(ones(segment_count))

#     # Generate values for each segment
#     segments = [{(:segments, i)} ~ normal(0, 1) for i=1:segment_count]
    
#     # Determine a global noise level (gamma-distributed)
#     z ~ normal(0, 1)  # z is Normally distributed
#     noise = quantile(Gamma(1, 1), cdf(Normal(0, 1), z))  # Transform z to noise using the gamma CDF
    
#     # Generate the y points for the input x points
#     xmin, xmax = extrema(xs)
#     cumfracs = cumsum(fractions)
#     cumfracs[end] = 1.0  # Ensure the last cumulative fraction is exactly 1.0

#     inds = [findfirst(frac -> frac >= (x - xmin) / (xmax - xmin), cumfracs) for x in xs]
#     segment_values = segments[inds]
#     for (i, val) in enumerate(segment_values)
#         {(:y, i)} ~ normal(val, noise)
#     end
# end

# # Generalized Elliptical Slice Sampling (ESS)
# function elliptical_slice_sampling(tr, model, args, observations, addresses, prior_dict)
#     # Current parameter value(s) for the selected addresses
#     current_values = [get_choices(tr)[addr] for addr in addresses]

#     # Extract prior mean (m0) and variance (t2) for each selected address
#     prior_means = []
#     prior_vars = []
#     for addr in addresses
#         # Get the prior distribution for the selected address
#         prior_dist = prior_dict[addr]
#         # Extract mean and variance from the prior distribution
#         if prior_dist isa Normal
#             push!(prior_means, prior_dist.μ)
#             push!(prior_vars, prior_dist.σ^2)
#         else
#             error("Prior distribution for $addr is not Normal. ESS requires Normal priors.")
#         end
#     end

#     # Standard deviations for the priors
#     prior_stds = sqrt.(prior_vars)

#     # Center the current values so the priors are ~N(0, prior_std^2)
#     current_values_offset = current_values .- prior_means

#     # Sample a random "direction" from the same priors: N(0, prior_std^2)
#     nu = prior_stds .* randn(length(current_values))

#     # Current log probability (includes prior + likelihood)
#     logp_cur = Gen.get_score(tr)

#     # Slice threshold: uniformly pick a threshold below logp_cur
#     logy = logp_cur + log(rand())

#     # Draw an initial angle
#     theta = 2π * rand()
#     # Bracket is [theta - 2π, theta]
#     theta_min = theta - 2π
#     theta_max = theta

#     # Elliptical slice loop
#     while true
#         # Proposed offset via "rotation"
#         proposed_offset = current_values_offset .* cos(theta) + nu .* sin(theta)
#         # Shift back by prior means
#         proposed_values = prior_means .+ proposed_offset

#         # Create a ChoiceMap with the proposed values
#         temp_cm = Gen.choicemap()
#         for (i, addr) in enumerate(addresses)
#             temp_cm[addr] = proposed_values[i]
#         end

#         # Attempt to update the trace with the new values
#         (temp_tr, _, re_score) = Gen.update(tr, args, (), temp_cm)

#         # Ensure re_score is a numeric type
#         re_score = Float64(Gen.get_score(temp_tr))

#         if re_score > logy
#             # Accept the proposal
#             return temp_tr
#         else
#             # Shrink the bracket
#             if theta < 0
#                 theta_min = theta
#             else
#                 theta_max = theta
#             end
#             # Sample a new angle within the bracket
#             theta = rand() * (theta_max - theta_min) + theta_min
#         end
#     end
# end

# # Function to compute log likelihood
# function compute_log_likelihood(traces)
#     log_likelihoods = [get_score(tr) for tr in traces]
#     mean_log_likelihood = logmeanexp(log_likelihoods)
#     return mean_log_likelihood
# end

# # Function to run inference
# function run_inference(method, model, args, observations, addresses, prior_dict, num_samples)
#     traces = []
#     (tr, _) = generate(model, args, observations)
#     for iter=1:num_samples
#         if method == :ess
#             tr = elliptical_slice_sampling(tr, model, args, observations, addresses, prior_dict)
#         elseif method == :mh
#             (tr, _) = mh(tr, select(addresses...))
#         elseif method == :hmc
#             (tr, _) = hmc(tr, select(addresses...))
#         end
#         push!(traces, tr)
#     end
#     return traces
# end

# # Example usage
# function main()
#     # Define the model arguments
#     xs = collect(range(0, stop=10, length=100))  # Input x values
#     args = (xs,)  # Arguments for the piecewise_constant model

#     # Define the observations (constraints)
#     ys = sin.(xs) .+ randn(length(xs)) * 0.1  # Simulated y values
#     ys = max.(ys, 0.0)  # Ensure ys is non-negative

#     observations = Gen.choicemap()
#     for (i, y) in enumerate(ys)
#         observations[(:y, i)] = y
#     end

#     # Define the addresses of variables to update
#     addresses = [:z, (:segments, 1), (:segments, 2)]  # Update z and segments

#     # Define the prior dictionary using addresses as keys
#     prior_dict = Dict(
#         :z => Normal(0, 1),  # Prior for z (transformed noise)
#         (:segments, 1) => Normal(0, 1),  # Prior for segment 1
#         (:segments, 2) => Normal(0, 1)   # Prior for segment 2
#     )

#     # Run inference
#     method = :ess  # Use Elliptical Slice Sampling
#     num_samples = 10000
#     traces = run_inference(method, piecewise_constant, args, observations, addresses, prior_dict, num_samples)

#     # Compute log likelihood
#     mean_log_likelihood = compute_log_likelihood(traces)
#     println("Mean Log Likelihood: ", mean_log_likelihood)

#     # Plot traces for z
#     z_samples = [get_choices(tr)[:z] for tr in traces]
#     plot(z_samples, xlabel="Iteration", ylabel="z", label="z", title="Trace Plot for z")

#     # Plot traces for segments
#     segment_samples = [[get_choices(tr)[(:segments, i)] for tr in traces] for i in 1:2]
#     plt = plot(layout=(2, 1), size=(800, 600))
#     for (i, seg) in enumerate(segment_samples)
#         plot!(plt[i], seg, xlabel="Iteration", ylabel="Segment $i", label="Segment $i")
#     end
#     display(plt)
# end

# # Run the main function
# main()


using Gen, Distributions, Plots, StatsPlots, Statistics, Distributions, LinearAlgebra

# Define the linear_regression_model
@gen function linear_regression_model(X, N, K, sigma_alpha2, mu_beta, sigma_beta2, lambda_sigma)
    alpha ~ normal(0, sqrt(sigma_alpha2))  # Intercept
    beta1 ~ normal(mu_beta, sqrt(sigma_beta2))  # Regression coefficient 1
    beta2 ~ normal(mu_beta, sqrt(sigma_beta2))  # Regression coefficient 2
    beta3 ~ normal(mu_beta, sqrt(sigma_beta2))  # Regression coefficient 3

    nu ~ gamma(2, 10)
    sigma ~ exponential(lambda_sigma)

    for i in 1:N
        mu_i = alpha + X[i, 1] * beta1 + X[i, 2] * beta2 + X[i, 3] * beta3
        {(:y, i)} ~ normal(mu_i, sigma)
    end
end

@gen function proposal()
    # Sample alpha and beta from a proposal distribution
    alpha ~ normal(0, 1)  # Proposal for alpha
    beta = [{(:beta, i)} ~ normal(0, 1) for i in 1:3]  # Proposal for beta
    return (alpha, beta)  # Return the proposed values
end

# Generalized Elliptical Slice Sampling (ESS)
function elliptical_slice_sampling(tr, model, args, observations, addresses, prior_dict)
    # Current parameter value(s) for the selected addresses
    current_values = [get_choices(tr)[addr] for addr in addresses]

    # Extract prior mean (m0) and variance (t2) for each selected address
    prior_means = []
    prior_vars = []
    for addr in addresses
        # Get the prior distribution for the selected address
        prior_dist = prior_dict[addr]
        # Extract mean and variance from the prior distribution
        if prior_dist isa Normal
            push!(prior_means, prior_dist.μ)
            push!(prior_vars, prior_dist.σ^2)
        else
            error("Prior distribution for $addr is not Normal. ESS requires Normal priors.")
        end
    end

    # Standard deviations for the priors
    prior_stds = sqrt.(prior_vars)

    # Center the current values so the priors are ~N(0, prior_std^2)
    current_values_offset = current_values .- prior_means

    # Sample a random "direction" from the same priors: N(0, prior_std^2)
    nu = prior_stds .* randn(length(current_values))

    # Current log probability (includes prior + likelihood)
    logp_cur = Gen.get_score(tr)

    # Slice threshold: uniformly pick a threshold below logp_cur
    logy = logp_cur + log(rand())

    # Draw an initial angle
    theta = 2π * rand()
    # Bracket is [theta - 2π, theta]
    theta_min = theta - 2π
    theta_max = theta

    # Elliptical slice loop
    while true
        # Proposed offset via "rotation"
        proposed_offset = current_values_offset .* cos(theta) + nu .* sin(theta)
        # Shift back by prior means
        proposed_values = prior_means .+ proposed_offset

        # Create a ChoiceMap with the proposed values
        temp_cm = Gen.choicemap()
        for (i, addr) in enumerate(addresses)
            temp_cm[addr] = proposed_values[i]
        end

        # Attempt to update the trace with the new values
        (temp_tr, _, re_score) = Gen.update(tr, args, (), temp_cm)

        # Ensure re_score is a numeric type
        re_score = Float64(Gen.get_score(temp_tr))

        if re_score > logy
            # Accept the proposal
            return temp_tr
        else
            # Shrink the bracket
            if theta < 0
                theta_min = theta
            else
                theta_max = theta
            end
            # Sample a new angle within the bracket
            theta = rand() * (theta_max - theta_min) + theta_min
        end
    end
end

# Function to compute log likelihood
function compute_log_likelihood(traces)
    log_likelihoods = [get_score(tr) for tr in traces]
    mean_log_likelihood = logmeanexp(log_likelihoods)
    return mean_log_likelihood
end

# # Function to run inference
# function run_inference(method, model, args, observations, addresses, prior_dict, num_samples)
#     traces = []
#     (tr, _) = generate(model, args, observations)
#     for iter=1:num_samples
#         if method == :ess
#             tr = elliptical_slice_sampling(tr, model, args, observations, addresses, prior_dict)
#         elseif method == :is
#             # Importance sampling generates a vector of traces
#             (new_traces, _) = importance_sampling(model, args, observations, proposal, (), num_samples)
#             # Append each trace to the traces array
#             for tr in new_traces
#                 push!(traces, tr)
#             end
#         elseif method == :mh
#             (tr, _) = mh(tr, select(addresses...))
#         elseif method == :hmc
#             (tr, _) = hmc(tr, select(addresses...))
#         end
#         push!(traces, tr)
#     end
#     return traces
# end

function run_inference(method, model, args, observations, addresses, prior_dict, num_samples)
    traces = []
    if method == :ess || method == :mh || method == :hmc
        (tr, _) = generate(model, args, observations)
        for iter=1:num_samples
            if method == :ess
                tr = elliptical_slice_sampling(tr, model, args, observations, addresses, prior_dict)
            elseif method == :mh
                (tr, _) = mh(tr, select(addresses...))
            elseif method == :hmc
                (tr, _) = hmc(tr, select(addresses...))
            end
            push!(traces, tr)
        end
    elseif method == :is
        # Importance sampling generates a vector of traces
        (new_traces, _) = importance_sampling(model, args, observations, proposal, (), num_samples)
        # Append each trace to the traces array
        for tr in new_traces
            push!(traces, tr)
        end
    end
    return traces
end

function logmeanexp(scores)
    logsumexp(scores) - log(length(scores))
end;

# Example usage
function main()
    # Define the model arguments
    N = 100  # Number of data points
    K = 3    # Number of predictors
    X = rand(N, K)  # Design matrix
    sigma_alpha2 = 1.0  # Prior variance for alpha
    mu_beta = 0.0       # Prior mean for beta
    sigma_beta2 = 1.0   # Prior variance for beta
    lambda_sigma = 1.0  # Rate parameter for sigma

    args = (X, N, K, sigma_alpha2, mu_beta, sigma_beta2, lambda_sigma)

    # Define the observations (constraints)
    ys = rand(N)  # Simulated y values
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # Define the addresses of variables to update
    addresses = [:alpha, :beta1, :beta2, :beta3]

    # Define the prior dictionary using addresses as keys
    prior_dict = Dict(
        :alpha => Normal(0, sqrt(sigma_alpha2)),
        :beta1 => Normal(mu_beta, sqrt(sigma_beta2)),
        :beta2 => Normal(mu_beta, sqrt(sigma_beta2)),
        :beta3 => Normal(mu_beta, sqrt(sigma_beta2))
    )

    # Run inference for all methods
    methods = [:hmc, :ess]
    results = Dict()
    for method in methods
        println("Running $method...")
        @time traces = run_inference(method, linear_regression_model, args, observations, addresses, prior_dict, 1000)
        results[method] = traces
    end

    # Compare log likelihood
    for method in methods
        mean_log_likelihood = compute_log_likelihood(results[method])
        println("Log Likelihood ($method): ", mean_log_likelihood)
    end

    # Plot traces for alpha
    plt = plot(layout=(2, 2), size=(800, 600))
    for (i, method) in enumerate(methods)
        alpha_samples = [get_choices(tr)[:alpha] for tr in results[method]]
        plot!(plt[i], alpha_samples, xlabel="Iteration", ylabel="alpha", label=string(method), title=string(method))
    end
    display(plt)
end

# Run the main function
main()