abstract type InferenceTarget end

struct MarginalDensityTarget <: InferenceTarget
    x::Float64
end

function cf(target::MarginalDensityTarget, t)
    cf(Normal(target.x), t)
end

#struct 1 <: InferenceTarget
#    x::Float64
#end
