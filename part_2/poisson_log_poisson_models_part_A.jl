# using Pkg
# # Uncomment to install if needed:
# # Pkg.add("Gen")
# # Pkg.add("Distributions")
# # Pkg.add("StatsPlots")
# # Pkg.add("Plots")
import Random, Logging
using Plots
using Gen
using Distributions
using StatsPlots



@gen function log_poisson(m0, t2)

    # Draw log(theta) from Normal(m0, sqrt(t2)):
    logtheta ~ normal(m0, sqrt(t2))
    # Draw y from Poisson(exp(logtheta)):
    y ~ poisson(exp(logtheta))
    return y
end

@gen function poissons(m0, t2)

    # Draw log(theta) from Normal(m0, sqrt(t2)):
    logtheta ~ normal(m0, sqrt(t2))
    # Draw y from Poisson(exp(logtheta)):
    y ~ poisson(logtheta)
    return y
end


function make_constraints(ys)
    constraints = Gen.choicemap()
    constraints[:y] = ys
    return constraints
end;

function logmeanexp(scores)
    logsumexp(scores) - log(length(scores))
end;


function elliptical_slice_1d(tr,
                             model,
                             m0,
                             t2)

    # Current parameter value
    x_cur = get_choices(tr)[:logtheta]
    # Standard deviation for the prior:
    prior_std = sqrt(t2)
    # Center it so prior is ~N(0, prior_std^2)
    x_offset_cur = x_cur - m0

    # Sample a random "direction" from the same prior:
    # i.e. from N(0, prior_std^2)
    nu = prior_std * randn()

    # Current log probability
    logp_cur = Gen.get_score(tr)  # includes prior + likelihood

    # Slice threshold
    logy = logp_cur + log(rand())  # uniformly picking a threshold below logp_cur

    # Draw an initial angle
    theta = 2π * rand()
    # Bracket is [theta - 2π, theta]
    theta_min = theta - 2π
    theta_max = theta

    # Elliptical slice loop
    while true
        # Proposed offset via "rotation"
        x_offset_prop = x_offset_cur * cos(theta) + nu * sin(theta)
        # Shift back by m0
        x_prop = m0 + x_offset_prop

        # Attempt to update trace with the new x_prop
        # "update" will re-simulate if needed and compute the new log score.
        # We'll keep the same (m0, t2) arguments.
        temp_cm = Gen.choicemap((:logtheta => x_prop))
        (temp_tr, _, re_score) = Gen.update(
            tr, 
            (m0, t2), (),   # arguments to the generative function
            temp_cm
        )

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


# Define the log_poisson model
@gen function log_poisson(m0, t2)
    logtheta ~ normal(m0, sqrt(t2))  # Prior on logtheta
    y ~ poisson(exp(logtheta))       # Poisson likelihood
    return y
end

# Define a proposal distribution for importance sampling
@gen function proposal(m0, t2)
    logtheta ~ normal(m0, sqrt(t2))  # Same as the prior
end

# Parameters
m0 = log(50.0)  # Prior for logtheta: log(50) ≈ 3.91
t2 = 1.0      # Prior variance (now on log scale)
ys = 60.0       # Observed data
obs = Gen.choicemap((:y => ys))  # Constraints
num_samples = 20000  # Number of samples

# Function to run inference with a given sampler
function run_inference(method, model, args, observations, num_samples)
    traces = []
    if method == :ess || method == :mh || method == :hmc
        (tr, _) = generate(model, args, observations)
        for iter=1:num_samples
            if method == :ess
                tr = elliptical_slice_1d(tr, model, args[1], args[2])
            elseif method == :mh
                (tr, _) = mh(tr, select(:logtheta))
            elseif method == :hmc
                (tr, _) = hmc(tr, select(:logtheta))
            end
            push!(traces, tr)
        end
    elseif method == :is
        # Importance sampling generates a vector of traces
        (new_traces, _) = importance_sampling(model, args, observations, proposal, args, num_samples)
        # Append each trace to the traces array
        for tr in new_traces
            push!(traces, tr)
        end
    end
    return traces
end

# Run inference for each method
# methods = [:ess, :mh, :is, :hmc]
methods = [:ess, :mh, :hmc, :is]
results = Dict()
for method in methods
    println("Running $method...")
    @time traces = run_inference(method, log_poisson, (m0, t2), obs, num_samples) #Change log_poisson => poissons to switch models
    results[method] = traces
end

# Plot trace plots for comparison
plt = plot(layout=(2, 2), size=(800, 600))
for (i, method) in enumerate(methods)
    # Extract logtheta values from each trace
    logtheta_samples = [get_choices(tr)[:logtheta] for tr in results[method]]
    # Plot the trace
    # density!(plt[i], logtheta_samples, xlabel="logtheta", ylabel="density", label=string(method), title=string(method))
    plot!(plt[i], logtheta_samples, xlabel="Iteration", ylabel="logtheta", label=string(method), title=string(method))
end
display(plt)

# Compare log probability estimates
for method in methods
    scores = [get_score(tr) for tr in results[method]]
    println("Log probability ($method): ", logmeanexp(scores))
end