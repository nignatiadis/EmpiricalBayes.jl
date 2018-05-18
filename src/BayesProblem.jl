# for now only univariate problems and with discretization

struct NormalConvolutionProblem
    prior::Distribution
    marginal::Vector{Float64} # measure after truncation & binning
    marginal_grid::Vector{Float64}
    marginal_h::Float64
end

struct DiscretizedNormalConvolutionProblem
    prior::Vector{Float64}
    prior_grid::Vector{Float64}
    prior_h::Float64
    marginal::Vector{Float64} # measure after truncation & binning
    marginal_grid::Vector{Float64}
    marginal_h::Float64
end

function marginal_grid_l(d::T) where T<:Union{NormalConvolutionProblem, DiscretizedNormalConvolutionProblem}
    d.marginal_grid .- d.marginal_h./2
end

function marginal_grid_r(d::T) where T<:Union{NormalConvolutionProblem, DiscretizedNormalConvolutionProblem}
    d.marginal_grid .+ d.marginal_h./2
end

function NormalConvolutionProblem(prior, marginal_grid)
    marginal_grid = collect(marginal_grid)
    marginal_h = marginal_grid[2] - marginal_grid[1]
    d = NormalConvolutionProblem(prior, zeros(Float64, length(marginal_grid)), marginal_grid, marginal_h)
    marginal_l = marginal_grid_l(d)
    marginal_r = marginal_grid_r(d)

    Z = Normal()

    f(x) = pdf(prior,x[1])*pdf(Z, x[2]-x[1]) #x[1] is the mean

    for i=1:length(marginal_l)
        # TODO: Fix hardcoded 20
       d.marginal[i] = hcubature(f, [-20, marginal_l[i]], [+20, marginal_r[i]])[1]
    end
    d.marginal ./= sum(d.marginal)
    d
end

function convolution_matrix(::Type{DiscretizedNormalConvolutionProblem}, prior_grid, marginal_grid; σ=1.0)
    #marginal_h = marginal_grid[2] - marginal_grid[1]
    A = reshape([pdf(Normal(0,σ), x-μ) for x in prior_grid for μ in marginal_grid],
                length(marginal_grid), length(prior_grid))#.*marginal_h
    A
end

function DiscretizedNormalConvolutionProblem(prior, prior_grid, marginal_grid)
    prior_grid = collect(prior_grid)
    marginal_grid = collect(marginal_grid)

    prior_h = prior_grid[2]-prior_grid[1]

    marginal_h = marginal_grid[2]-marginal_grid[1]
    A = convolution_matrix(DiscretizedNormalConvolutionProblem,
                                prior_grid, marginal_grid) # should this be stored maybe?

    f_marginal = A*prior
    f_marginal ./= sum(f_marginal)

    DiscretizedNormalConvolutionProblem(prior, prior_grid, prior_h,
                             f_marginal, marginal_grid, marginal_h)
end


function DiscretizedNormalConvolutionProblem(prior_distr::Distribution, prior_grid, marginal_grid)
    prior_h = prior_grid[2]-prior_grid[1]
    prior = pdf.(prior_distr, prior_grid).*prior_h # makes this an approximate prob. measure
    DiscretizedNormalConvolutionProblem(prior, prior_grid, marginal_grid)
end


#---- TODO: maybe deprecate these two functions eventually
function posterior_stats(d::NormalConvolutionProblem, f, x)
    prior = d.prior
    Z = Normal()
    post_num =  quadgk(β ->  f(β)*pdf(prior, β)*pdf(Z, x-β) , -20,20)[1]
    post_denom = quadgk(β ->  pdf(prior, β)*pdf(Z, x-β) , -20,20)[1]
    (post_num, post_denom, post_num/post_denom)
end

function posterior_stats(d::DiscretizedNormalConvolutionProblem, ψ, x)
    prior = d.prior
    prior_grid = d.prior_grid
    post_num = d.prior'*(ψ.(prior_grid).*pdf.(Normal(), prior_grid .- x))
    post_denom = d.prior'* pdf.(Normal(), prior_grid .- x)
    (post_num, post_denom, post_num/post_denom)
end
#----------------------------------------------------------------------
# main fun -- rename eventually
function posterior_stats(d::NormalConvolutionProblem, t::LinearInferenceTarget)
    prior = d.prior
    quadgk(β ->  pdf(prior, β)*riesz_representer(t, β) , -20,20)[1]
end

function posterior_stats(d::NormalConvolutionProblem, t::PosteriorTarget)
    num = posterior_stats(d, t.num)
    denom = posterior_stats(d, t.denom)
    (num, denom, num/denom)
end



function StatsBase.rand(d::NormalConvolutionProblem, n)
     μs = rand(d.prior, n)
     μs .+ randn(n)
end

# standardize if we mean marginal or prior pdf?
function pdf(d::NormalConvolutionProblem, x)
    posterior_stats(d, MarginalDensityTarget(x))
end


struct MixingNormalConvolutionProblem{T<:Distribution}
    priors::Vector{T}
    prior_mixture_coef::Vector{Float64}
    marginal_map::Matrix{Float64} #map pi_s -> marginal dbn
    marginal_grid::Vector{Float64}
    marginal_h::Float64
end


function MixingNormalConvolutionProblem(priors::Vector{T},
            marginal_grid,
            prior_mixture_coef=ones(length(priors))./length(priors)) where T<:Distribution

    marginal_grid = collect(marginal_grid)

    #marginal as function of specified prior..
    n_priors = length(priors)
    n_marginal_grid = length(marginal_grid)
    marginal_h = marginal_grid[2] - marginal_grid[1]

    marginal_map = zeros(Float64, n_marginal_grid, n_priors)
    for i=1:n_priors
        tmp = NormalConvolutionProblem(priors[i], marginal_grid)
        marginal_map[:,i] = tmp.marginal
    end
    MixingNormalConvolutionProblem(priors, ones(n_priors)./n_priors,
                    marginal_map, marginal_grid, marginal_h)
end

function MixingNormalConvolutionProblem(::Type{Normal}, σ, prior_grid, marginal_grid)
    priors = [Normal(μ, σ) for μ in prior_grid]
    MixingNormalConvolutionProblem(priors, marginal_grid)
end

# TODO: Implement using iterator specification instead..
function index_mixing(d::MixingNormalConvolutionProblem, i)
    NormalConvolutionProblem(d.priors[i], d.marginal_grid)
end


function posterior_stats(d::MixingNormalConvolutionProblem, t)
    [posterior_stats(index_mixing(d,i), t) for i=1:length(d.priors)]
end

function Base.length(ds::MixingNormalConvolutionProblem)
    length(ds.priors)
end

function Distributions.pdf(ds::MixingNormalConvolutionProblem, mixing_coeff, grid)
     d = MixtureModel(ds.priors, vec(mixing_coeff))
     pdf.(d, grid)
end

function Distributions.pdf(ds::MixingNormalConvolutionProblem, grid)
     Distributions.pdf(ds, ds.prior_mixture_coef, grid)
end

function NormalConvolutionProblem(ds::MixingNormalConvolutionProblem, mixing_coeff)
    prior =  MixtureModel(ds.priors, vec(mixing_coeff))
    NormalConvolutionProblem(prior, ds.marginal_grid)
end
