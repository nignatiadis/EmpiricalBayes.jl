struct DonohoCI
    est_num::Float64
    est_denom::Float64
    est_target::Float64
    calibrated_target::Float64
    ci_left::Float64
    ci_right::Float64
    f::BinnedMarginalDensity
end

function donoho_ci(Xs, marginal_grid, prior_grid, target_x, σ, est_method)
    n_total = length(Xs)
    n_half = ceil(Int, n_total/2)
    idx_test = sample(1:n_total, n_half, replace=false)
    idx_train = setdiff(1:n_total, idx_test)
    Xs_train = Xs[idx_train]
    Xs_test = Xs[idx_test]

    # Train
    #Estimated density + BinnedMarginalDensity
    (est_num, est_denom, f) = myfit(Xs_train, target_x, marginal_grid, est_method, σ)
    est_target = max(0,min(1,est_num/est_denom))

    # numerator/denominator
    ds = MixingNormalConvolutionProblem(Normal, σ, prior_grid, marginal_grid)
    target_f = x-> 1*(x>=0) - est_target
    # Test: Use the Donoho calibrator on the learned function
    ma = MinimaxCalibrator(ds, f, n_half, target_f, target_x; rel_tol=1e-5)

    QXs =  ma.(Xs_test)
    sd = std(QXs)/sqrt(n_half)
    max_bias = ma.max_bias
    zz =  get_plus_minus(max_bias, sd)

    zz = zz/est_denom
    calib_target = est_target + mean(QXs)/est_denom
    ci_left = calib_target - zz
    ci_right = calib_target + zz

    DonohoCI(est_num,
        est_denom,
        est_target,
        calib_target,
        ci_left,
        ci_right,
        f)
end

function myfit(Xs, target_x, marginal_grid, est_method::NormalConvolutionProblem, σ)
    d_true = est_method
    post_stats = posterior_stats(d_true, x->1*(x>=0), target_x)
    f = BinnedMarginalDensity(d_true)
    (post_stats[1], post_stats[2], f)
end

function myfit(Xs, target_x, marginal_grid, est_method::Type{NPMLE}, σ)
    prior_grid = linspace(-5,5, 401)
    npmle = fit(NPMLE, collect(prior_grid), collect(marginal_grid), Xs; σ=σ)
    d_npmle = NormalConvolutionProblem(npmle, marginal_grid)
    post_stats = posterior_stats(d_npmle, x->1*(x>=0), target_x)
    f = BinnedMarginalDensity(d_npmle)
    (post_stats[1], post_stats[2], f)
end
