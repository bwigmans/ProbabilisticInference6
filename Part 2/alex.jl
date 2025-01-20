using Gen
using Distributions

"""
    multi_armed_bandit_model(alpha, beta, chosen_arms)

Gen model for a Bayesian Multi-Armed Bandit using Beta-Bernoulli arms.

# Arguments
- `alpha::Vector{Float64}`: A vector of α parameters for the Beta priors (size M).
- `beta::Vector{Float64}`: A vector of β parameters for the Beta priors (size M).
- `chosen_arms::Vector{Int}`: The sequence of chosen arms over T trials.

# Returns
A Gen model that samples:
1. θᵢ ~ Beta(αᵢ, βᵢ) for i=1..M
2. rₜ ~ Bernoulli(θᵣ) depending on the arm chosen at trial t
"""
@gen function multi_armed_bandit_model(alpha::Vector{Float64},
                                       beta::Vector{Float64},
                                       chosen_arms::Vector{Int})
    M = length(alpha)      # Number of arms
    T = length(chosen_arms)  # Number of trials

    # 1) Sample each arm's probability of reward.
    thetas = Vector{Float64}(undef, M)
    for i in 1:M
        thetas[i] = @trace(beta(alpha[i], beta[i]), (:theta, i))
    end

    # 2) For each trial, we observe a reward from the chosen arm.
    for t in 1:T
        arm_t = chosen_arms[t]           # Which arm was chosen at time t
        @trace(bernoulli(thetas[arm_t]), (:reward, t))
    end
end


"""
    hmm_model(pi, A, B, T)

Gen model for a discrete Hidden Markov Model (HMM).

# Arguments
- `pi::Vector{Float64}`:   Initial state distribution (size N).
- `A::Matrix{Float64}`:    State transition matrix (N x N).
- `B::Matrix{Float64}`:    Emission matrix (N x K). Rows sum to 1.
- `T::Int`:                Number of time steps.

# Returns
A Gen model that samples:
1. s₁ ~ Categorical(π)
2. sₜ ~ Categorical(A[sₜ₋₁, :]) for t=2..T
3. oₜ ~ Categorical(B[sₜ, :])    for t=1..T
"""
@gen function hmm_model(pi::Vector{Float64},
                        A::Matrix{Float64},
                        B::Matrix{Float64},
                        T::Int)
    # 1) Sample the initial hidden state.
    s1 = @trace(categorical(pi), (:state, 1))

    # 2) Sample subsequent hidden states based on transitions from A.
    prev_state = s1
    for t in 2:T
        st = @trace(categorical(A[prev_state, :]), (:state, t))
        prev_state = st
    end

    # 3) For each time step, sample an emission from the state-dependent distribution in B.
    for t in 1:T
        current_state = get_choices($(Gen.dotget(:state, t))).value
        @trace(categorical(B[current_state, :]), (:obs, t))
    end
end