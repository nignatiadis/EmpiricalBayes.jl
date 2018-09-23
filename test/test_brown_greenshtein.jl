using EmpiricalBayes
using Test

using Distributions

marginal_grid = collect(range(-6, stop=6, length=1001));
true_dist = Normal(0,1)
d_true = NormalConvolutionProblem(true_dist, marginal_grid);

m = 7_000
Xs = rand(d_true, m)

estimate(Xs, BrownGreenshtein, PosteriorTarget(PosteriorMeanNumerator(2.0)))
posterior_stats(d_true, PosteriorTarget(PosteriorMeanNumerator(2.0)))

estimate(Xs, BrownGreenshtein, PosteriorTarget(PosteriorMeanNumerator(4.0)))
posterior_stats(d_true, PosteriorTarget(PosteriorMeanNumerator(4.0)))

estimate(Xs, BrownGreenshtein, PosteriorTarget(PosteriorMeanNumerator(3.0)))
posterior_stats(d_true, PosteriorTarget(PosteriorMeanNumerator(3.0)))
