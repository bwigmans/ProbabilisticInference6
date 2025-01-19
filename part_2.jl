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
    constraints
end;

function logmeanexp(scores)
    logsumexp(scores) - log(length(scores))
end;

function block_resimulation_update(tr)
    (tr, _) = mh(tr, select(:logtheta))
    (tr, _) = hmc(tr, select(:logtheta))

    tr
end;

function block_resimulation_inference(m0, t2, ys, observations)
    observations = make_constraints(ys)
    num_samples = 5000
    (tr, _) = generate(log_poisson, (m0, t2), observations)
    for iter=1:num_samples
        tr = block_resimulation_update(tr)
    end
    (traces, log_norm_weights, lml_est) = importance_sampling(log_poisson, (m0, t2), observations, num_samples)
    return tr
end

ys = 48.0
m0 = 50.0
t2 = 10.0
scores = Vector{Float64}(undef, 10)
for i=1:10
    @time tr = block_resimulation_inference(m0, t2, ys, observations)
    scores[i] = get_score(tr)
end
println("Log probability: ", logmeanexp(scores))