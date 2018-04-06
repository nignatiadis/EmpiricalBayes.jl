struct NPMLE{T<:Distribution}
    prior_grid::Vector{Float64}
    prior_mixing::Vector{Float64}
    marginal_grid::Vector{Float64}
    marginal::T
    σ::Float64
end


function npmle_dual_hist(prior_grid, marginal_grid, Xs;  θ=0.0, maxiter=1_000_000, σ_std=1.0)

    h = marginal_grid[2] - marginal_grid[1]
    edges = [marginal_grid-h/2; maximum(marginal_grid)+h/2]
    edges[1] = -Inf
    edges[end] = Inf

    hist = fit(Histogram, Xs, edges; closed=:right)

    idx = find( hist.weights .> 0)

    marginal_grid_active = marginal_grid[idx]
    #A = NormalConvolutionMatrix(prior_grid,Xs)
    A = convolution_matrix(DiscretizedNormalConvolutionProblem, prior_grid, marginal_grid_active; σ=σ_std);

    L = real(tsvd(A)[2][1])
    σ = 1/L/1.1
    τ = 1/L/1.1

    N = sum(hist.weights)
    m = length(prior_grid)

    wts = hist.weights[idx]

    x_n = wts./pdf.(Normal(0,std(Xs)),marginal_grid_active)# ν we are interested in
    x_n_tmp = copy(x_n)

    y_n = zeros(prior_grid)
    y_n[ceil(Int,m/2)] = 1.0
    y_n_tmp = copy(y_n)

    bar_x_n = copy(x_n)


    function prox_σFstar(x, σ)
        x - σ*max(0,min(x/σ, N))
    end

    function prox_τG(y, ct, τ )
        (y + sqrt(y^2 + 4τ*ct))/2
    end

    for i=1:maxiter
       At_mul_B!(y_n_tmp, A, bar_x_n)
       y_n .+= σ.*y_n_tmp
       y_n .= prox_σFstar.(y_n, σ)

       bar_x_n .= x_n
       idx = find(y_n .> 0)


       A_mul_B!(x_n_tmp, view(A, :, idx), view(y_n, idx))
       x_n .= x_n .- τ.*x_n_tmp
       x_n .= prox_τG.(x_n, wts, τ)

       bar_x_n .= (1+θ).*x_n .- θ.*bar_x_n

       if mod(i-1,10000) == 0
           if i>=20_000 && (abs(sum(y_n)-1) <= 0.01) && norm(x_n_tmp .- wts./x_n, 2) <= 0.025
                break
           end
       #    @show sum(h.*wts./x_n[1:end])
        end
    end

    #A_full=convolution_matrix(DiscretizedNormalConvolutionProblem, prior_grid, marginal_grid);

    #(y_n, A_full*y_n)
    y_n
end


function StatsBase.fit(::Type{NPMLE}, prior_grid, marginal_grid, Xs; σ=0.0)
    σ_std = sqrt(σ^2 + 1)
   res= npmle_dual_hist(prior_grid, marginal_grid, Xs; σ_std=σ_std)
   # add some checks that everything went well..
   idx = find(res .> 0)
   prior_grid = prior_grid[idx]
   prior_mixing = res[idx]
   prior_mixing = prior_mixing/sum(prior_mixing)
   m = MixtureModel(Normal, collect(zip(prior_grid, σ*ones(prior_grid))), prior_mixing)
   NPMLE(prior_grid, prior_mixing, marginal_grid, m, σ)
end


function MixingNormalConvolutionProblem(a::NPMLE, marginal_grid)
    priors = a.marginal.components
    mixing_coef = a.marginal.prior.p
    MixingNormalConvolutionProblem(priors ,marginal_grid, mixing_coef)
end


function NormalConvolutionProblem(a::NPMLE, marginal_grid)
   m =  MixingNormalConvolutionProblem(a ,marginal_grid)
   NormalConvolutionProblem(m, a.prior_mixing)
end
