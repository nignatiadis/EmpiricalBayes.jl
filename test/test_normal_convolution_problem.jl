using EmpiricalBayes
using Test
using Distributions


## test whether speed-up for normal distributions indeed works
# i.e. whether it is faster and numerically works
marginal_grid = collect(range(-6, stop=6, length=1001));

# same distribution, but once as Normal type, and once as mixture type..
dist1= MixtureModel([ Normal(0,.2), Normal(0,.2)])
dist2= Normal(0,.2)


normal_conv_prob1 = NormalConvolutionProblem(dist1, marginal_grid)
normal_conv_prob2 = NormalConvolutionProblem(dist2, marginal_grid)

@test normal_conv_prob1.marginal ≈ normal_conv_prob2.marginal

t1 = @elapsed NormalConvolutionProblem(dist1, marginal_grid)
t2 = @elapsed NormalConvolutionProblem(dist2, marginal_grid)

@test t1 > t2
