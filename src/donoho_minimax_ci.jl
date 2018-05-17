struct DonohoCI
    est_num::Float64
    est_denom::Float64
    est_target::Float64
    calibrated_target::Float64
    ci_left::Float64
    ci_right::Float64
    std::Float64
    max_bias::Float64
    f::BinnedMarginalDensity
    ma::MinimaxCalibrator
end

# provides lightweight CI functionality for linear functionals
function donoho_ci(Xs, ma::MinimaxCalibrator; conf=0.9)
    est = mean(ma.Q.(Xs))
    sd = std(ma.Q.(Xs))/sqrt(length(Xs))
    max_bias = ma.max_bias
    zz =  get_plus_minus(max_bias, sd, conf)
    (est, est-zz, est+zz)
end


function donoho_ci(Xs, marginal_grid, prior_grid, σ, target::PosteriorTarget, est_method)
    n_total = length(Xs)
    n_half = ceil(Int, n_total/2)
    idx_test = sample(1:n_total, n_half, replace=false)
    idx_train = setdiff(1:n_total, idx_test)
    Xs_train = Xs[idx_train]
    Xs_test = Xs[idx_test]
    donoho_ci(Xs_train, Xs_test, marginal_grid, prior_grid, σ, target, est_method)
end



function donoho_ci(Xs_train, Xs_test, marginal_grid, prior_grid, σ, target::PosteriorTarget, est_method)
    # fix this
    n_total = length(Xs_test) + length(Xs_train)
    n_half = length(Xs_test)
    # Train
    #Estimated density + BinnedMarginalDensity
    (est_num, est_denom, f) = myfit(Xs_train, target, marginal_grid, est_method, σ)
    est_target = max(0,min(1,est_num/est_denom))

    # numerator/denominator
    ds = MixingNormalConvolutionProblem(Normal, σ, prior_grid, marginal_grid)

    calib_target = CalibratedNumerator(target.num, est_target)
    # Test: Use the Donoho calibrator on the learned function
    ma = MinimaxCalibrator(ds, f, n_half, calib_target; tol=1e-5)

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
        sd/est_denom,
        max_bias/est_denom,
        f,
        ma)
end

function myfit(Xs, target, marginal_grid, est_method::NormalConvolutionProblem, σ)
    d_true = est_method
    post_stats = posterior_stats(d_true, target)
    f = BinnedMarginalDensity(d_true)
    (post_stats[1], post_stats[2], f)
end

function myfit(Xs, target, marginal_grid, est_method::Type{NPMLE}, σ)
    prior_grid = linspace(-5,5, 401)
    npmle = fit(NPMLE, collect(prior_grid), collect(marginal_grid), Xs; σ=σ)
    d_npmle = NormalConvolutionProblem(npmle, marginal_grid)
    post_stats = posterior_stats(d_npmle, target)
    f = BinnedMarginalDensity(d_npmle)
    (post_stats[1], post_stats[2], f)
end

# Alternative signature

#function donoho_test2(Xs_train, Xs_test, est_num, est_denom, f::BinnedMarginalDensity,
#            ds::MixingNormalConvolutionProblem, target::PosteriorTarget; ε=M_bd, C=Inf, conf=0.9)

function donoho_test2(Xs_train, Xs_test,  f::BinnedMarginalDensity,
            ds::MixingNormalConvolutionProblem, target::PosteriorTarget; C=Inf, conf=0.9, kwargs...)
    # fix this
    n_half = length(Xs_test)

    # Train
    # Estimate the numerator
    M_max_num = MinimaxCalibrator(ds, f, length(Xs_train), target.num;
                 C=C, kwargs...);

    num_res = donoho_ci(Xs_train, M_max_num; conf=conf)

    # Estimate the denominator
    M_max_denom = MinimaxCalibrator(ds, f, length(Xs_train), target.denom;
                C=C, kwargs...);

    denom_res = donoho_ci(Xs_train, M_max_denom; conf=conf)

    #TODO: Check if denominator not sure to be >0... abort or throw warning
    est_target = num_res[1]/denom_res[1]

    calib_target = CalibratedNumerator(target.num, est_target)
    # Test: Use the Donoho calibrator on the learned function
    ma = MinimaxCalibrator(ds, f, n_half, calib_target; C=C, kwargs...)

    QXs =  ma.(Xs_test)
    sd = std(QXs)/sqrt(n_half)
    max_bias = ma.max_bias
    zz =  get_plus_minus(max_bias, sd, conf)

    zz = zz/denom_res[1]
    calib_target = est_target + mean(QXs)/denom_res[1]
    ci_left = calib_target - zz
    ci_right = calib_target + zz

    don_ci = DonohoCI(num_res[1],
        denom_res[1],
        est_target,
        calib_target,
        ci_left,
        ci_right,
        sd/denom_res[1],
        max_bias/denom_res[1],
        f,
        ma)

    return (don_ci, M_max_num, M_max_denom)
end