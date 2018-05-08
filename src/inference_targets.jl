abstract type InferenceTarget end

abstract type LinearInferenceTarget <: InferenceTarget end

struct MarginalDensityTarget <: LinearInferenceTarget
    x::Float64
end

function cf(target::MarginalDensityTarget, t)
    cf(Normal(target.x), t)
end

function riesz_representer(target::MarginalDensityTarget, t)
    pdf(Normal(), target.x - t)
end

# running under assumption X_i >=0...
struct LFSRNumerator <: LinearInferenceTarget
    x::Float64
end

function cf(target::LFSRNumerator, t)
    x = target.x
    exp(im*t*x- t^2/2)*(1+im*erfi((t-im*x)/sqrt(2)))/2
end

function riesz_representer(target::LFSRNumerator, t)
    pdf(Normal(), target.x - t)*(t>=0)
end

#---------Should probably just turn this all into one function-----
struct CalibratedNumerator <: LinearInferenceTarget
    num::LinearInferenceTarget
    θ̄::Float64 #Pilot
end

function riesz_representer(target::CalibratedNumerator, t)
    x = num.x
    riesz_representer(num.x -t) - riesz_representer(MarginalDensityTarget(x),t)
end


#------------------- Beyond linear functionals ------------------------

struct PosteriorTarget <: InferenceTarget
    num::LinearInferenceTarget
    denom::MarginalDensityTarget
end

function PosteriorTarget(lfsr::LFSRNumerator)
    x = lfsr.x
    PosteriorTarget(lfsr, MarginalDensityTarget(x))
end
