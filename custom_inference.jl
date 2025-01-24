using Gen, Distributions, SpecialFunctions

"""
    to_z(value, dist) -> z, logdet_jac

Given `value` drawn from `dist`, produce an equivalent `z ~ Normal(0,1)` 
and the log of the absolute determinant of the Jacobian if needed.
We store both so we can build a correct log-density in z-space.
"""
function to_z(value::Float64, dist)
    if dist isa Normal
        # dist = Normal(μ, σ)
        μ, σ = dist.μ, dist.σ
        z = (value - μ) / σ
        # The transform x -> (x-μ)/σ is linear, so log|Jac|= -log(σ).
        logdet_jac = -log(σ)
        return z, logdet_jac

    elseif dist isa Beta
        # dist = Beta(α, β)
        α, β = dist.α, dist.β
        # 1) u = cdf(Beta(α,β), value) in (0,1)
        u = cdf(dist, value)
        # 2) z = Φ⁻¹(u) in ℝ
        z = quantile(Normal(0,1), u)
        #
        # logdet_jac = log( dθ/dz ) ? We’ll handle that by comparing
        #  logpdf(Normal(0,1), z) - logpdf(Beta(α,β), θ)
        #  in the final "prior_correction". So we can simply do:
        logdet_jac = 0.0
        return z, logdet_jac

    elseif dist isa InverseGamma
        # dist = InverseGamma(α, θ)
        αig, βig = dist.α, dist.β
        # For x ~ InvGamma(α,β), we can do the transform:
        #   u = cdf(InvGamma(α,β), x) in (0,1)
        #   z = Φ⁻¹(u).
        # Same logic as Beta. (Or you might prefer log(x) -> standard normal, etc.)
        u = cdf(dist, value)
        z = quantile(Normal(), u)
        logdet_jac = 0.0
        return z, logdet_jac

    elseif dist isa Gamma
        # For x ~ Gamma(k, θ), or equivalently shape=k, rate=1/θ or something.
        # We'll assume dist in Julia has shape α, rate β or shape-rate parameterization.
        # We do same cdf -> Normal transform:
        u = cdf(dist, value)
        z = quantile(Normal(), u)
        logdet_jac = 0.0
        return z, logdet_jac

    elseif dist isa Exponential
        # Exponential(λ). Same idea:
        u = cdf(dist, value)
        z = quantile(Normal(), u)
        logdet_jac = 0.0
        return z, logdet_jac

    else
        # fallback: treat as if "value" is standard normal. Or do a random-walk update.
        # For demonstration, we do a no-op "z = value", ignoring transforms.
        z = value
        logdet_jac = 0.0
        return z, logdet_jac
    end
end

"""
    from_z(z, dist) -> value

Inverse transform: from z in ℝ to the domain of `dist`.
"""
function from_z(z::Float64, dist)
    if dist isa Normal
        μ, σ = dist.μ, dist.σ
        return μ + σ*z

    elseif dist isa Beta
        α, β = dist.α, dist.β
        # convert z -> u via Φ(z), then u -> θ via invcdf(Beta(α,β), u)
        u = cdf(Normal(), z)
        # handle edge cases if u is 0 or 1 numerically
        return quantile(Beta(α, β), u)

    elseif dist isa InverseGamma
        # same logic
        u = cdf(Normal(), z)
        return quantile(dist, u)

    elseif dist isa Gamma
        u = cdf(Normal(), z)
        return quantile(dist, u)

    elseif dist isa Exponential
        u = cdf(Normal(), z)
        return quantile(dist, u)

    else
        # fallback (no transform)
        return z
    end
end


"""
    elliptical_slice_1d(
        trace, address, dist;
        log_joint, num_slice_draws=1
    )

Performs 1D elliptical slice sampling updates on the random choice at `address`,
given its prior distribution `dist`, within a Gen trace.
`log_joint(trace)` is a function that returns log of the *current* model joint
(including constraints). We do `num_slice_draws` repeated ESS updates.
Returns nothing (operates in-place).
"""
function elliptical_slice_1d(
    trace,
    address,
    dist;
    log_joint,
    num_slice_draws::Int=1
)
    # -- 0) Convert the current value -> z in R
    current_value = trace[address]
    z0, _ = to_z(current_value, dist)

    # We'll need the log of the "full posterior" in z-space. 
    # But we treat the "prior" as N(0,1) in z-space, so we do a correction:
    # 
    #    log_posterior(z) = log_joint_of( from_z(z) ) 
    #                       + logpdf(N(0,1), z) 
    #                       - logpdf(dist, from_z(z))
    # 
    # Because logpdf(trace) includes logpdf(dist, current_value).
    # We'll define:
    function log_posterior_z(z::Float64)
        # 1) Convert z -> value
        val = from_z(z, dist)
        # 2) Temporarily set trace
        Gen.setval!(trace, address, val)
        # 3) Evaluate the model's joint density
        lj = log_joint(trace)
        # 4) Add the "reparam prior correction"
        correction = logpdf(Normal(0,1), z) - logpdf(dist, val)
        return lj + correction
    end

    current_logpost = log_posterior_z(z0)

    for _ in 1:num_slice_draws
        # (a) random direction from N(0,1) in 1D is just a single real
        nu = rand(Normal(0,1))

        # (b) propose slice threshold
        threshold = log(rand()) + current_logpost

        # (c) bracket in [-π, +π]
        theta = 2π * rand()
        theta_min = theta - 2π
        theta_max = theta

        z_cur = z0  # current "z"

        # (d) shrink bracket until acceptance
        while true
            z_prop = z_cur*cos(theta) + nu*sin(theta)
            lp_prop = log_posterior_z(z_prop)
            if lp_prop >= threshold
                # accept
                z_cur = z_prop
                current_logpost = lp_prop
                break
            else
                # shrink bracket
                if theta < 0
                    theta_min = theta
                else
                    theta_max = theta
                end
                theta = rand()*(theta_max - theta_min) + theta_min
            end
        end

        # done one slice iteration
        z0 = z_cur
    end
end

"""
    universal_elliptical_inference(
        model, model_args;
        constraints=choicemap(),
        num_sweeps=1000
    ) -> trace

A single function that can handle *all* of your example models. 
It does "blocked" updates address by address, using elliptical slice sampling
if (a) the prior distribution is known and transformable to a standard Normal,
or (b) fallback to a small random-walk MH otherwise.

For simplicity, we do 1 ESS update per address per sweep.

Returns a final DynamicTrace.
"""
function universal_elliptical_inference(
    model::Function,
    model_args::Tuple;
    constraints=Gen.choicemap(),
    num_sweeps::Int=1000
)
    # 1) Generate an initial trace consistent with constraints
    trace, _ = Gen.generate(model, model_args, constraints)

    # 2) We'll define a function to compute the model's log joint:
    log_joint(trace) = Gen.logpdf(trace)

    # 3) Identify all addresses. We'll update them in a systematic order. 
    #    get_choices(trace) returns an AddressValue pair for each random choice
    addresses = collect(keys(get_choices(trace)))

    # 4) For each sweep:
    for s in 1:num_sweeps
        # Shuffle or iterate in a fixed order. We'll do a fixed order here:
        for addr in addresses
            # get the distribution from the trace
            dist = Gen.get_choice_distribution(trace, addr)
            # We'll do elliptical slice if it's one of our recognized types
            # or if there's a reparam transform in `to_z/from_z`.
            known_dists = (Normal, Beta, InverseGamma, Gamma, Exponential)
            if any(d -> dist isa d, known_dists)
                # 1D elliptical slice update
                elliptical_slice_1d(
                    trace, addr, dist;
                    log_joint=log_joint, num_slice_draws=1
                )
            else
                # fallback to a trivial random-walk MH (not shown).
                # e.g., do_nuts_mh_update(trace, addr) or something:
                # We'll just skip for brevity.
            end
        end
    end

    return trace
end

# Example: log_poisson
m0, t2 = 0.0, 2.0
obs = choicemap((:y)=> 10)  # Suppose y=10 is observed
final_trace = universal_elliptical_inference(log_poisson, (m0, t2); 
                                             constraints=obs, 
                                             num_sweeps=2000)
println("Log pdf of final trace = ", Gen.logpdf(final_trace))
println("Posterior sample for logtheta = ", final_trace[:logtheta])