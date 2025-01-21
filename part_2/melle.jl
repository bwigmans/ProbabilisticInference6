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
    period = ({:period} ~ gamma(1, 1))
    phase = ({:phase} ~ uniform(0, 2*pi))

    function sine(x)
        return amplitude * sin(((2*pi*x)/period) + phase)
    end
    noise = ({:noise} ~ gamma(1, 1))
    for (i, x) in enumerate(xs)
        {(:y, i)} ~ normal(sine(x), noise)
    end
    return y
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

traces = [simulate(piecewise_constant, (xs_dense,)) for _ in 1:9]
plot([visualize_trace(t) for t in traces]...)
