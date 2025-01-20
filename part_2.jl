using Pkg
# Uncomment to install if needed:
# Pkg.add("Gen")
# Pkg.add("Distributions")
# Pkg.add("StatsPlots")
# Pkg.add("Plots")
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
    for i=1:length(ys)
        constraints[:y] = ys
    end
    return constraints
end;

function logmeanexp(scores)
    logsumexp(scores) - log(length(scores))
end;

# -------------------------------------------------------------------
# 1D Elliptical Slice Sampler for :logtheta
# -------------------------------------------------------------------
"""
    elliptical_slice_1d(tr, model, m0, t2; selection = select(:logtheta))

Performs a single elliptical slice sampling update on the one-dimensional
latent variable `:logtheta`. Assumes a Normal(m0, sqrt(t2)) prior.

Returns the updated trace.
"""
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

function block_resimulation_update(tr, model, (), obs)
    # (tr, _) = mh(tr, select(:logtheta))
    # (tr, _) = hmc(tr, select(:logtheta))
    # tr
end;


function block_resimulation_inference(m0, t2, ys, observations)
    num_samples = 5000
    (tr, _) = generate(poissons, (m0, t2), observations)
    for iter=1:num_samples
        tr = elliptical_slice_1d(tr, poissons, m0, t2)
    end

    return tr
end

ys = 48.0
m0 = 50.0
t2 = 10.0
obs = make_constraints(ys)
scores = Vector{Float64}(undef, 10)
for i=1:10
    @time tr = block_resimulation_inference(m0, t2, ys, obs)
    scores[i] = get_score(tr)
end
println("Log probability: ", logmeanexp(scores))