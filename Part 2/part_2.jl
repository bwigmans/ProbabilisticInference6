using Pkg
# Uncomment if needed:
# Pkg.add("Gen")
# Pkg.add("Distributions")
# Pkg.add("StatsPlots")

using Gen
using Distributions
using StatsPlots



@gen function my_model(m0, t2)
    # Draw logtheta from Normal(m0, sqrt(t2))
    logtheta = @trace(normal(m0, sqrt(t2)), :logtheta)

    # Draw y from Poisson(exp(logtheta)), labeled as :y
    y = @trace(poisson(exp(logtheta)), :y)

    return y
end

function my_inference_program(m0::Float64, t2::Float64, y_obs::Float64, num_iters::Int)
    # Create a set of constraints for the observed y
    constraints = choicemap()
    constraints[:y] = y_obs

    # Generate an initial trace from the model with y constrained
    (trace, _) = generate(my_model, (m0, t2), constraints)

    # Metropolis–Hastings on :logtheta only (m0 and t2 are fixed)
    for _ in 1:num_iters
        (trace, _) = metropolis_hastings(trace, select(:logtheta))
    end

    # Extract the final logtheta
    final_choices = get_choices(trace)
    final_logtheta = final_choices[:logtheta]
    return final_logtheta
end

# Example usage:
m0    = 1.0   # prior mean
t2    = 0.5   # prior variance
y_obs = 1.0   # observed Poisson count
num_iters = 1000

inferred_logtheta = my_inference_program(m0, t2, y_obs, num_iters)
println("Inferred logtheta = $inferred_logtheta")
