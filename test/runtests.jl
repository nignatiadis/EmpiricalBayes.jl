using EmpiricalBayes
using Base.Test
using Distributions
using QuadGK

# Move these to other parts eventually

# test if convolution matrix works for small n..
A = convolution_matrix(DiscretizedNormalConvolutionProblem, linspace(-1,1,4), linspace(-7,7,2000))
h = linspace(-7,7,2000)[2] - linspace(-7,7,2000)[1]
@test isapprox(sum(A,1)*h, ones(4)'; atol=1e-4)


# test discretization approach vs. properly integrating..
n_marginal_grid = 1001
marginal_grid =  linspace(-3,3, n_marginal_grid)
marginal_h = marginal_grid[2]-marginal_grid[1]

n_prior_grid = 1001
prior_grid = linspace(-6,6, n_prior_grid)

normal_normal = DiscretizedNormalConvolutionProblem(Normal(0,0.5), 
                    prior_grid,
                    marginal_grid)

marginal_dist = Normal(0,sqrt(1+0.5^2))
marginal_l = marginal_grid .- marginal_h/2
marginal_r = marginal_grid .+ marginal_h/2

f_marginal = [QuadGK.quadgk(x->pdf.(marginal_dist,x),marginal_l[i],marginal_r[i])[1] for i=1:n_marginal_grid];
f_marginal ./= sum(f_marginal);

@test isapprox(f_marginal, normal_normal.marginal; atol= 1e-7)

@test sum(normal_normal.prior) ≈ 1
@test sum(normal_normal.marginal) ≈ 1

# check if posterior mean calculated correctly (normal-normal conjugate we know this ofc)
post_mean = posterior_stats(normal_normal, μ -> μ, 2)
@test isapprox(post_mean[3], 2/1/(1+1/0.5^2))


# now check for indicator μ >= 0
# actually tolerances here are rather bad...esp. for ground truth should do a better job.

true_num = quadgk(β ->  pdf(Normal(0,0.5), β)*pdf(Normal(), 2-β) , 0,100)[1]
true_denom = quadgk(β ->  pdf(Normal(0,0.5), β)*pdf(Normal(), 2-β) , -100,100)[1]
true_ratio = true_num/true_denom

posterior_positive = posterior_stats(normal_normal, μ ->  (μ >= 0).*1, 2)
@test isapprox(true_num, posterior_positive[1]; atol=1e-3 )
@test isapprox(true_denom, posterior_positive[2]; atol=1e-3 )
@test isapprox(true_ratio, posterior_positive[3]; atol=1e-2)


# again in this case we know exactly what the value is..

post_Normal = Normal(2/1/(1+1/0.5^2), sqrt(1*0.5^2/(1+0.5^2)))
post_positive_true = ccdf(post_Normal, 0 )

@test isapprox(post_positive_true, true_ratio)
@test isapprox(post_positive_true, posterior_positive[3]; atol=1e-2)


## OK NOW let us check with the more exact object

normal_normal_exact = NormalConvolutionProblem(Normal(0,0.5), marginal_grid)
@test isapprox(f_marginal, normal_normal_exact.marginal; atol= 1e-9)


non_disr_post_positive = posterior_stats(normal_normal_exact, μ ->  (μ >= 0).*1, 2)
@test isapprox(post_positive_true, non_disr_post_positive[3]; atol=1e-8)
