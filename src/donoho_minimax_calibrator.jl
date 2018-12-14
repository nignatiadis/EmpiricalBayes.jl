struct BinnedCalibrator
    marginal_grid::Vector{Float64}
    marginal_h::Float64
    Q::Vector{Float64}
    Qo::Float64
end

function BinnedCalibrator(marginal_grid, Q, Qo)
    marginal_h = marginal_grid[2] - marginal_grid[1]
    BinnedCalibrator(marginal_grid, marginal_h, Q, Qo)
end

function BinnedCalibrator(marginal_grid, Q)
    BinnedCalibrator(marginal_grid, Q, zero(Float64))
end


function (c::BinnedCalibrator)(x)
    marginal_grid = c.marginal_grid
    marginal_h = c.marginal_h
    Q = c.Q
    # TODO: Some checks that we don't violate stuff..
    # Vectorized version of this..?
    idx = searchsortedlast(marginal_grid .- marginal_h/2, x)
    idx = max(1, min(idx, length(marginal_grid)))
    Q[idx] + c.Qo
end

abstract type LinearEstimator end

mutable struct MinimaxCalibrator
    Q::BinnedCalibrator
    max_bias::Float64
    sd::Float64 # not to be trusted though (i.e. recalculate)
    f1::Vector{Float64}
    f2::Vector{Float64}
    π1::Vector{Float64}
    π2::Vector{Float64}
    ds::MixingNormalConvolutionProblem
    f::BinnedMarginalDensity #have separate f for neighborhood and calibration?
    m::Int64
    target::LinearInferenceTarget
    ε_reg::Float64
    C::Float64
    δ::Float64
end

function (c::MinimaxCalibrator)(x)
    (c.Q)(x)
end


function MinimaxCalibrator(ds::MixingNormalConvolutionProblem,
                  f::BinnedMarginalDensity, m,
                  target = LFSRNumerator(2.0);
                  C=0.1,
                  max_iter=300,
                  ε = isinf(C) ? 0.001 : C,
                  tol=1e-5,
                  bias_check=false,
                  solver = GurobiSolver(OutputFlag=0))


    n_priors = length(ds.priors)

    n_marginal_grid = length(ds.marginal_grid)
    h_marginal_grid = ds.marginal_grid[2] - ds.marginal_grid[1]

    f_marginal = f.marginal
    f_marginal_reg = max.(f.marginal,ε*h_marginal_grid)


    # Let us get linear functional
    #post_stats = posterior_stats(ds, target)
    #L = [z[1] for z in post_stats]
    L = posterior_stats(ds, target)

    A = ds.marginal_map;

    jm = Model(solver=solver)
    @variable(jm, π1[1:n_priors] >= 0)
    @variable(jm, π2[1:n_priors] >= 0)
    @variable(jm, t)
    @variable(jm, f1[1:n_marginal_grid])
    @variable(jm, f2[1:n_marginal_grid])

    @constraint(jm, sum(π1)== 1 )
    @constraint(jm, sum(π2)== 1 )
    @constraint(jm, A*π1 .== f1)
    @constraint(jm, A*π2 .== f2)

    if (C < Inf)
        #@constraint(jm, f1 .- f_marginal .<= C*f_marginal_reg)
        #@constraint(jm, f1 .- f_marginal .>= -C*f_marginal_reg)
        #@constraint(jm, f2 .- f_marginal .<= C*f_marginal_reg)
        #@constraint(jm, f2 .- f_marginal .>= -C*f_marginal_reg)
        @constraint(jm, f1 .- f_marginal .<= C*h_marginal_grid)
        @constraint(jm, f1 .- f_marginal .>= -C*h_marginal_grid)
        @constraint(jm, f2 .- f_marginal .<= C*h_marginal_grid)
        @constraint(jm, f2 .- f_marginal .>= -C*h_marginal_grid)
    end

    @objective(jm, Max, dot(L,π1-π2))
    status = solve(jm)

    t_curr = getobjectivevalue(jm)
    K_squared_curr = norm(sqrt(m)*(getvalue(f1).-getvalue(f2))./sqrt.(f_marginal_reg), 2)^2

    curr_obj = t_curr^2/(K_squared_curr+4)


    t_min = 0.0;
    t_max = t_curr;
    t_mid = (t_min + t_curr)/2;

    @constraint(jm, bias, dot(L,π1-π2) == t_mid)
    # update objective as well
    @objective(jm, Min, sum((f1-f2).^2 ./f_marginal_reg))
    status = solve(jm)

    for j=1:max_iter
        deriv = (2*t_mid*(4+getobjectivevalue(jm)*m)- getdual(bias)*t_mid^2*m)/(4+getobjectivevalue(jm)*m)^2

        if (deriv >= 0)
            t_min = t_mid
        else
            t_max = t_mid
        end
        t_mid = (t_min + t_max)/2

        JuMP.setRHS( bias, t_mid)
        status = solve(jm)

        curr_obj = t_mid^2/(4+getobjectivevalue(jm)*m)

        (t_max - t_min < tol) && break # actually could easily calibrate at beginning
    end

    fg1 = getvalue(f1)
    fg2 = getvalue(f2)

    K_squared = getobjectivevalue(jm)*m
    δ = sqrt(getobjectivevalue(jm))

    tg = t_mid

    π1 = getvalue(π1)
    π2 = getvalue(π2)

    L_g1 = dot(L, π1)
    L_g2 = dot(L, π2)
    L_g0 = (L_g1 + L_g2)/2

    Q = vec(m/(4+K_squared).*(fg1.-fg2)./f_marginal_reg .* tg)
    # ok the other bias term which I thought was going to 0...
    Qo = - 0.5*tg*sum( (fg1.^2 .- fg2.^2)./f_marginal_reg)
    Qo = Qo/(4 + K_squared)*m

    Qo = L_g0 + Qo

    max_bias = 2*tg/(4+K_squared)
    sd = tg*sqrt(K_squared)/(4+K_squared)
    #(max_bias, sd, max_bias^2+sd^2, next_obj)


    Q_c = BinnedCalibrator(ds.marginal_grid, Q, Qo)



    # Let us recalculate the max bias to be on the safe side?

    ma = MinimaxCalibrator(Q_c,
                      max_bias,
                      sd,
                      fg1,
                      fg2,
                      π1,
                      π2,
                      ds,
                      f,
                      m,
                      target,
                      ε,
                      C,
                      δ )

    if bias_check
        opt1 = check_bias(ma, solver=solver)
        opt2 = check_bias(ma; maximization= false, solver=solver)
        max_bias = max(abs(opt1), abs(opt2))
        ma.max_bias = max_bias
    end

    ma

end


function check_bias(Q::BinnedCalibrator,
                  ds::MixingNormalConvolutionProblem,
                  f::BinnedMarginalDensity, m,
                  target = LFSRNumerator(2.0);
                  C=Inf, maximization=true,
                  solver=GurobiSolver(OutputFlag=0))


    n_priors = length(ds.priors)

    f_marginal = f.marginal

    n_marginal_grid = length(ds.marginal_grid)

    h_marginal_grid = ds.marginal_grid[2] - ds.marginal_grid[1]

    A = ds.marginal_map;
    post_stats = posterior_stats(ds, target)
    L = [z[1] for z in post_stats]

    jm = Model(solver=solver)

    @variable(jm, π3[1:n_priors] >= 0)
    @variable(jm, f3[1:n_marginal_grid])

    @constraint(jm, sum(π3)== 1 )
    @constraint(jm, A*π3 .== f3)

    if (C < Inf)
        @constraint(jm, f3 .- f_marginal .<= C*h_marginal_grid)
        @constraint(jm, f3 .- f_marginal .>= -C*h_marginal_grid)
    end

    if maximization
        @objective(jm, Max, Q.Qo + dot(Q.Q,f3)-dot(L,π3))
    else
        @objective(jm, Min, Q.Qo + dot(Q.Q,f3)-dot(L,π3))
    end

    status = solve(jm)

    t_curr = getobjectivevalue(jm)
end


function check_bias(ma::MinimaxCalibrator; maximization=true)
    ds = ma.ds
    f = ma.f
    C = ma.C
    ε = ma.ε_reg
    m = ma.m
    target = ma.target

    Q = ma.Q

    check_bias(Q, ds, f, m, target; C=C, maximization=maximization)
end
