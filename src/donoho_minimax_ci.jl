struct CalibratedCI
    est_num::Float64
    est_num_linear::EmpiricalBayesEstimator #LinearEstimator
    est_denom::Float64
    est_denom_linear::EmpiricalBayesEstimator #LinearEstimator
    est_target::Float64
    calibrated_target::Float64
    ci_left::Float64
    ci_right::Float64
    std::Float64
    max_bias::Float64
    f::BinnedMarginalDensityNeighborhood
    ma::MinimaxCalibrator
end

function estimate(L::LinearEstimator, Xs)
    mean(L.(Xs))
end

function estimate(L::LinearEstimator, target::LinearInferenceTarget, Xs)
    # TODO: Add check that target is correct target
    mean(L.(Xs))
end
# provides lightweight CI functionality for linear functionals
function donoho_ci(Xs, ma::MinimaxCalibrator; conf=0.9)
    est = mean(ma.Q.(Xs))
    sd = std(ma.Q.(Xs))/sqrt(length(Xs))
    max_bias = ma.max_bias
    zz =  get_plus_minus(max_bias, sd, conf)
    (est, est-zz, est+zz)
end



function CEB_ci(Xs_train, Xs_test,
            ds::MixingNormalConvolutionProblem,
            target::PosteriorTarget,
            M_max_num::EmpiricalBayesEstimator, #
            M_max_denom::EmpiricalBayesEstimator=M_max_num;
            C=:auto, conf=0.9, kwargs...)


    m_train = length(Xs_train)
    m_test = length(Xs_test)

    marginal_grid = ds.marginal_grid
    marginal_h = ds.marginal_h


    num_res = estimate(M_max_num, target.num, Xs_train)
    denom_res = estimate(M_max_denom, target.denom, Xs_train)

    #TODO: Check if denominator not sure to be >0... abort or throw warning
    est_target = num_res/denom_res

    calib_target = CalibratedNumerator(target.num, est_target)
            # Test: Use the Donoho calibrator on the learned function

    #TODO : Change to KWarg..
    f_nb = fit(BinnedMarginalDensityNeighborhood, Xs_train, ds)

    if C==:auto
        C = f_nb.C_std*(1+f_nb.η_infl) + f_nb.C_bias
    end

    ma = MinimaxCalibrator(ds, f_nb.f, m_test, calib_target; C=C, kwargs...)

    QXs =  ma.(Xs_test)
    sd = std(QXs)/sqrt(m_test)
    max_bias = ma.max_bias
    zz =  get_plus_minus(max_bias, sd, conf)

    zz = zz/denom_res
    calib_target = est_target + mean(QXs)/denom_res
    ci_left = calib_target - zz
    ci_right = calib_target + zz

    don_ci = CalibratedCI(num_res,
                M_max_num,
                denom_res,
                M_max_denom,
                est_target,
                calib_target,
                ci_left,
                ci_right,
                sd/denom_res[1],
                max_bias/denom_res[1],
                f_nb,
                ma)

    return don_ci
end

function CEB_ci(Xs_train, Xs_test,
            ds::MixingNormalConvolutionProblem,
            target::PosteriorTarget;
            C=:auto,
            conf=0.9,
            kwargs...)
    # fix this
    m_train = length(Xs_train)
    m_test = length(Xs_test)

    marginal_grid = ds.marginal_grid
    marginal_h = ds.marginal_h

    M_bd = marginal_h/sqrt(2*pi)

    f_const = BinnedMarginalDensity(M_bd, marginal_grid, marginal_h)
    # Train
    # Estimate the numerator
    M_max_num = MinimaxCalibrator(ds, f_const, m_train, target.num;
                 C=Inf, kwargs...);


    # Estimate the denominator
    M_max_denom = MinimaxCalibrator(ds, f_const, m_train, target.denom;
                C=Inf, kwargs...);


    CEB_ci(Xs_train, Xs_test, ds, target,
                M_max_num,
                M_max_denom;
                C=C, conf=conf, kwargs...)

end

# TODO: Delete this, instead clean up
function CEB_ci_cb(Xs_train, Xs_test,
            ds::MixingNormalConvolutionProblem, target::PosteriorTarget, C_bias;
            C=:auto, conf=0.9, kwargs...)

            m_train = length(Xs_train)
            m_test = length(Xs_test)

            marginal_grid = ds.marginal_grid
            marginal_h = ds.marginal_h

            cb_num = ComteButucea( target.num, length(Xs_train), marginal_grid)
            cb_denom = ComteButucea( target.denom, length(Xs_train), marginal_grid)

            num_res = estimate(Xs_train, ComteButucea, target.num, marginal_grid)
            denom_res = estimate(Xs_train, ComteButucea, target.denom, marginal_grid)

            if denom_res <= 0
                denom_res = 1e-10
                num_res =0
            end
                    #TODO: Check if denominator not sure to be >0... abort or throw warning
            est_target = num_res[1]/denom_res[1]

            calib_target = CalibratedNumerator(target.num, est_target)
                    # Test: Use the Donoho calibrator on the learned function

            #TODO : Change to KWarg..
            f_nb = fit(BinnedMarginalDensityNeighborhood, Xs_train, marginal_grid)
            f_nb.C_bias = C_bias

            if C==:auto
                C = f_nb.C_std + f_nb.C_bias
            end

            ma = MinimaxCalibrator(ds, f_nb.f, m_test, calib_target; C=C, kwargs...)

            QXs =  ma.(Xs_test)
            sd = std(QXs)/sqrt(m_test)
            max_bias = ma.max_bias
            zz =  get_plus_minus(max_bias, sd, conf)

            zz = zz/denom_res[1]
            calib_target = est_target + mean(QXs)/denom_res[1]
            ci_left = calib_target - zz
            ci_right = calib_target + zz

            don_ci = CalibratedCI(num_res[1],
                        cb_num,
                        denom_res[1],
                        cb_denom,
                        est_target,
                        calib_target,
                        ci_left,
                        ci_right,
                        sd/denom_res[1],
                        max_bias/denom_res[1],
                        f_nb,
                        ma)

            return don_ci
end


function StatsBase.confint(ci::CalibratedCI, target::PosteriorTarget)
    (ci.ci_left, ci.ci_right)
end

function estimate(ci::CalibratedCI, target::PosteriorTarget)
    ci.calibrated_target
end
