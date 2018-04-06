struct NPMLE{T<:Distribution}
    prior_grid::Vector{Float64}
    prior_mixing::Vector{Float64}
    marginal_grid::Vector{Float64}
    marginal::T
end


function npmle_dual_hist(prior_grid, marginal_grid, Xs;  θ=0.0, maxiter=1_000_000)

    h = marginal_grid[2] - marginal_grid[1]
    edges = [marginal_grid-h/2; maximum(marginal_grid)+h/2]
    edges[1] = -Inf
    edges[end] = Inf

    hist = fit(Histogram, Xs, edges; closed=:right)

    idx = find( hist.weights .> 0)

    marginal_grid_active = marginal_grid[idx]
    #A = NormalConvolutionMatrix(prior_grid,Xs)
    A = convolution_matrix(DiscretizedNormalConvolutionProblem, prior_grid, marginal_grid_active);

    L = real(tsvd(A)[2][1])
    σ = 1/L/1.1
    τ = 1/L/1.1

    N = sum(hist.weights)
    m = length(prior_grid)

    wts = hist.weights[idx]

    x_n = wts./pdf.(Normal(0,std(Xs)),marginal_grid_active)# ν we are interested in
    x_n_tmp = copy(x_n)

    y_n = zeros(prior_grid)
    y_n_tmp = zeros(prior_grid)

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
       A_mul_B!(x_n_tmp, A, y_n)
       x_n .= x_n .- τ.*x_n_tmp
       x_n .= prox_τG.(x_n, wts, τ)

       bar_x_n .= (1+θ).*x_n .- θ.*bar_x_n

       if mod(i-1,10000) == 0
           if (abs(sum(y_n)-1) <= 0.01) && norm(A*y_n - wts./x_n, 2) <= 0.025
                break
           end
       #    @show sum(h.*wts./x_n[1:end])
        end
    end

    #A_full=convolution_matrix(DiscretizedNormalConvolutionProblem, prior_grid, marginal_grid);

    #(y_n, A_full*y_n)
    y_n
end


function fit(::Type{NPMLE}, prior_grid, marginal_grid, Xs)
   res= npmle_dual_hist(prior_grid, marginal_grid, Xs)
   # add some checks that everything went well..
   idx = find(res .> 0)
   prior_grid = prior_grid[idx]
   prior_mixing = res[idx]
   prior_mixing = prior_mixing/sum(prior_mixing)
   m = MixtureModel(Normal, prior_grid, prior_mixing)
   NPMLE(prior_grid, prior_mixing, marginal_grid, m)
end

