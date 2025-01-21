using Gen
using Distributions

@gen function multi_armed_bandit_model(
    α::Vector{Float64},
    β::Vector{Float64},
    chosen_arms::Vector{Int}
)
    M = length(α)
    T = length(chosen_arms)
    
    # Trace theta
    θ = [@trace(beta(α[i], β[i]), (:θ, i)) for i in 1:M]
    
    # rewards for arms
    for t in 1:T
        arm = chosen_arms[t]
        @trace(bernoulli(θ[arm]), (:reward, t))
    end
    
    return θ  # Return just θ for clarity
end

# 1. Define hyperparameters
α = [1.0, 1.0]  # Beta(1,1) priors for 2 arms
β = [1.0, 1.0]
chosen_arms = [1, 1, 1, 2] # Arm choices over 4 trials

# 2. Generate a trace (forward simulation)
trace = Gen.simulate(multi_armed_bandit_model, (α, β, chosen_arms))

# 3. Access theta
M = length(α)
sampled_θ = [trace[(:θ, i)] for i in 1:M]  # Access each arm's θ
println("Sampled θ: ", sampled_θ)

# 4. Access rewards
sampled_rewards = [trace[(:reward, t)] for t in 1:length(chosen_arms)]
println("Sampled rewards: ", sampled_rewards)