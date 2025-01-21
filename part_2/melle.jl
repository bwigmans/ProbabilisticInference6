import Random, Logging
using Gen, Plots, Logging, GenDistributions

Logging.disable_logging(Logging.Info);

@gen function line_model_fancy(xs::Vector{Float64})
    slope = ({:slope} ~ normal(0, 1))
    intercept = ({:intercept} ~ normal(0, 2))
    
    function y(x)
        return slope * x + intercept
    end
    
    noise = ({:noise} ~ gamma(1, 1))
    for (i, x) in enumerate(xs)
        {(:y, i)} ~ normal(slope * x + intercept, noise)
    end
    return y
end;

@gen function sine_model_fancy(xs::Vector{Float64})

    amplitude = ({:amplitude} ~ gamma(1, 1))
    period = ({:period} ~ gamma(5, 1))
    phase = ({:phase} ~ uniform(0, 2*pi))

    function sine(x)
        return amplitude * sin(((2*pi*x)/period) + phase)
    end
    noise = ({:noise} ~ gamma(1, 1))
    for (i, x) in enumerate(xs)
        {(:y, i)} ~ normal(sine(x), noise)
    end
    return sine
end;


@gen function combined_model(xs::Vector{Float64})
    if ({:is_line} ~ bernoulli(0.5))
        # Call line_model_fancy on xs, and import
        # its random choices directly into our trace.
        return ({*} ~ line_model_fancy(xs))
    else
        # Call sine_model_fancy on xs, and import
        # its random choices directly into our trace
        return ({*} ~ sine_model_fancy(xs))
    end
end;

function overlay(renderer, traces; same_data=true, args...)
    fig = renderer(traces[1], show_data=true, args...)
    
    xs, = get_args(traces[1])
    xmin = minimum(xs)
    xmax = maximum(xs)

    for i=2:length(traces)
        y = get_retval(traces[i])
        test_xs = collect(range(-5, stop=5, length=1000))
        fig = plot!(test_xs, map(y, test_xs), color="black", alpha=0.5, label=nothing,
                    xlim=(xmin, xmax), ylim=(xmin, xmax))
    end
    return fig
end;

function render_trace(trace; show_data=true)
    
    # Pull out xs from the trace
    xs, = get_args(trace)
    
    xmin = minimum(xs)
    xmax = maximum(xs)

    # Pull out the return value, useful for plotting
    y = get_retval(trace)
    
    # Draw the line
    test_xs = collect(range(-5, stop=5, length=1000))
    fig = plot(test_xs, map(y, test_xs), color="black", alpha=0.5, label=nothing,
                xlim=(xmin, xmax), ylim=(xmin, xmax))

    if show_data
        ys = [trace[(:y, i)] for i=1:length(xs)]
        
        # Plot the data set
        scatter!(xs, ys, c="black", label=nothing)
    end
    
    return fig
end;

function do_inference(model, xs, ys, amount_of_computation)
    
    # Create a choice map that maps model addresses (:y, i)
    # to observed values ys[i]. We leave :slope and :intercept
    # unconstrained, because we want them to be inferred.
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end
    
    # Call importance_resampling to obtain a likely trace consistent
    # with our observations.
    (trace, _) = Gen.importance_resampling(model, (xs,), observations, amount_of_computation);
    return trace
end;


xs = [-5., -4., -3., -2., -1., 0., 1., 2., 3., 4., 5.];
ys = [6.75003, 6.1568, 4.26414, 1.84894, 3.09686, 1.94026, 1.36411, -0.83959, -0.976, -1.93363, -2.91303];
ys_sine = [2.89, 2.22, -0.612, -0.522, -2.65, -0.133, 2.70, 2.77, 0.425, -2.11, -2.76];

# Run and visualize importance resampling on the combined regression model

traces = [do_inference(combined_model, xs, ys, 10000) for _=1:12];
linear_dataset_plot = overlay(render_trace, traces)
traces = [do_inference(combined_model, xs, ys_sine, 10000) for _=1:12];
sine_dataset_plot = overlay(render_trace, traces)

Plots.plot(linear_dataset_plot, sine_dataset_plot)



# Piecewise Constant model, using OOTB MH to solve gives decent results for simple and medium difficulty datasets

# A distribution that is guaranteed to be 1 or higher.
@dist poisson_plus_one(rate) = poisson(rate) + 1;

@gen function piecewise_constant(xs::Vector{Float64})
    # Generate a number of segments (at least 1)
    segment_count ~ poisson_plus_one(1)
    
    # To determine changepoints, draw a vector on the simplex from a Dirichlet
    # distribution. This gives us the proportions of the entire interval that each
    # segment takes up. (The entire interval is determined by the minimum and maximum
    # x values.)
    
    fractions ~ Gen.dirichlet(ones(segment_count))

    # Generate values for each segment
    segments = [{(:segments, i)} ~ normal(0, 1) for i=1:segment_count]
    
    # Determine a global noise level
    noise ~ gamma(1, 1)
    
    # Generate the y points for the input x points
    xmin, xmax = extrema(xs)
    cumfracs = cumsum(fractions)
    # Correct numeric issue: cumfracs[end] might be 0.999999
    @assert cumfracs[end] ≈ 1.0
    cumfracs[end] = 1.0

    inds = [findfirst(frac -> frac >= (x - xmin) / (xmax - xmin),
                      cumfracs) for x in xs]
    segment_values = segments[inds]
    for (i, val) in enumerate(segment_values)
        {(:y, i)} ~ normal(val, noise)
    end
end;

function trace_to_dict(tr)
    Dict(:values => [tr[(:segments, i)] for i=1:(tr[:segment_count])],
         :fracs  => tr[:fractions], :n => tr[:segment_count], :noise => tr[:noise],
         :ys => [tr[(:y, i)] for i=1:length(xs_dense)])
end;

function visualize_trace(tr; title="")
    xs, = get_args(tr)
    tr = trace_to_dict(tr)
    
    scatter(xs, tr[:ys], label=nothing, xlabel="X", ylabel="Y")
    
    cumfracs = [0.0, cumsum(tr[:fracs])...]
    xmin = minimum(xs)
    xmax = maximum(xs)
    for i in 1:tr[:n]
        segment_xs = [xmin + cumfracs[i] * (xmax - xmin), xmin + cumfracs[i+1] * (xmax - xmin)]
        segment_ys = fill(tr[:values][i], 2)
        plot!(segment_xs, segment_ys, label=nothing, linewidth=4)
    end
    Plots.title!(title)
end

xs_dense = collect(range(-5, stop=5, length=50));

# Below code shows randomly sampled traces from the piecewise constant model when other plots are commented out

# traces = [simulate(piecewise_constant, (xs_dense,)) for _ in 1:9]
# plot([visualize_trace(t) for t in traces]...)

NUM_CHAINS = 9
function make_constraints(ys)
    choicemap([(:y, i) => ys[i] for i=1:length(ys)]...)
end;

function visualize_mh_alg(xs, ys, update, frames=200, iters_per_frame=1, N=NUM_CHAINS; verbose=true)    
    traces = [first(generate(piecewise_constant, (xs,), make_constraints(ys))) for _ in 1:N]
    viz = Plots.@animate for i in 1:frames
        
        if i*iters_per_frame % 100 == 0 && verbose
            println("Iteration $(i*iters_per_frame)")
        end
        
        for j in 1:N
            for k in 1:iters_per_frame
                traces[j] = update(traces[j], xs, ys)
            end
        end
       
        Plots.plot([visualize_trace(t; title=(j == 2 ? "Iteration $(i*iters_per_frame)/$(frames*iters_per_frame)" : "")) for (j,t) in enumerate(traces)]...)#, layout=l)
    end
    scores = [Gen.get_score(t) for t in traces]
    println("Log mean score: $(logsumexp(scores) - log(N))")
    gif(viz)
end;

function simple_update(tr, xs, ys)
    tr, = mh(tr, select(:segment_count, :fractions))
    tr, = mh(tr, select(:fractions))
    tr, = mh(tr, select(:noise))
    for i=1:tr[:segment_count]
        tr, = mh(tr, select((:segments, i)))
    end
    tr
end

ys_simple  = ones(length(xs_dense)) .+ randn(length(xs_dense)) * 0.1
ys_medium  = Base.ifelse.(Int.(floor.(abs.(xs_dense ./ 3))) .% 2 .== 0,
                          2, 0) .+ randn(length(xs_dense)) * 0.1;
ys_complex = Int.(floor.(abs.(xs_dense ./ 2))) .% 5 .+ randn(length(xs_dense)) * 0.1;

# Run and visualize MH runs on the piecewise constant function

# visualize_mh_alg(xs_dense, ys_simple, simple_update, 100, 1)
visualize_mh_alg(xs_dense, ys_medium, simple_update, 50, 10)
# visualize_mh_alg(xs_dense, ys_complex, simple_update, 50, 10)

