abstract type InferenceTarget end

abstract type LinearInferenceTarget <: InferenceTarget end

abstract type PosteriorNumeratorTarget <: LinearInferenceTarget end


# TODO: Design this more carefully, maybe add function location() instead of requiring .x access


#-------- Marginal Density --------------------------------
struct MarginalDensityTarget <: LinearInferenceTarget
    x::Float64
end

function cf(target::MarginalDensityTarget, t)
    cf(Normal(target.x), t)
end

function riesz_representer(target::MarginalDensityTarget, t)
    pdf(Normal(), target.x - t)
end

pretty_label(target::MarginalDensityTarget) = L"f(x)"


#-------- Marginal Density --------------------------------
struct MarginalDistributionTarget <: LinearInferenceTarget
    x::Float64
end


function riesz_representer(target::MarginalDistributionTarget, t)
    cdf(Normal(), target.x - t)
end

pretty_label(target::MarginalDistributionTarget) = L"F(x)"



#----------- LFSRNumerator ---------------------------------
# running under assumption X_i >=0...
struct LFSRNumerator <: PosteriorNumeratorTarget
    x::Float64
end

function cf(target::LFSRNumerator, t)
    x = target.x
    exp(im*t*x- t^2/2)*(1+im*erfi((t-im*x)/sqrt(2)))/2
end

function riesz_representer(target::LFSRNumerator, t)
    pdf(Normal(), target.x - t)*(t>=0)
end

#--------- PosteriorMeanNumerator ---------------------------------

struct PosteriorMeanNumerator <: PosteriorNumeratorTarget
    x::Float64
end

function cf(target::PosteriorMeanNumerator, t)
    x = target.x
    cf(Normal(target.x), t)*(x + im*t)
end

function riesz_representer(target::PosteriorMeanNumerator, t)
    pdf(Normal(), target.x - t)*t
end

#--------------- GeneralPosteriorLinearTarget --------------------------

struct GeneralPosteriorLinearTarget <: PosteriorNumeratorTarget
    f::Function
    x::Float64
end

function riesz_representer(target::GeneralPosteriorLinearTarget, t)
    pdf(Normal(), target.x - t)*target.f(t)
end

#---------Should probably just turn this all into one function-----
struct CalibratedNumerator <: LinearInferenceTarget
    num::LinearInferenceTarget
    θ̄::Float64 #Pilot
end

function riesz_representer(target::CalibratedNumerator, t)
    x = target.num.x
    θ̄ = target.θ̄
    riesz_representer(target.num, t) - θ̄*riesz_representer(MarginalDensityTarget(x),t)
end
#----------------------------------------------------------------------

struct PriorTailProbability <: LinearInferenceTarget
    cutoff::Float64
end

function riesz_representer(target::PriorTailProbability, t)
    one(Float64)*(abs(t) >= target.cutoff)
end

#------------------- Beyond linear functionals ------------------------

struct PosteriorTarget{T} <: InferenceTarget where T<:LinearInferenceTarget
    num::T
    denom::MarginalDensityTarget
end

function PosteriorTarget(target::T) where T<:PosteriorNumeratorTarget
    x = target.x
    PosteriorTarget{T}(target, MarginalDensityTarget(x))
end

pretty_label(target::PosteriorTarget{LFSRNumerator}) = L"\Pr[\mu_i \geq 0 \mid X_i=x]"
pretty_label(target::PosteriorTarget{PosteriorMeanNumerator}) = L"E[\mu_i \mid X_i=x]"
