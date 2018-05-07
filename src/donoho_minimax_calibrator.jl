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


mutable struct MinimaxCalibrator
    Q::BinnedCalibrator
    max_bias::Float64
    sd::Float64 # not to be trusted though (i.e. recalculate)
    f1::Vector{Float64}
    f2::Vector{Float64}
    π1::Vector{Float64}
    π2::Vector{Float64}
    ds::MixingNormalConvolutionProblem
    f::BinnedMarginalDensity
    m::Int64
    target_f::Function
    target_x::Float64
    ε_reg::Float64
    C::Float64
end

function (c::MinimaxCalibrator)(x)
    (c.Q)(x)
end


function MinimaxCalibrator(ds::MixingNormalConvolutionProblem,
                  f::BinnedMarginalDensity, m,
                  target_f = x->1*(x>=0), target_x=2.0;
                  C=2.0, max_iter=300,ε = 1e-6, rel_tol=1e-5, bias_check=true)


    n_priors = length(ds.priors)

    f_marginal = f.marginal
    f_marginal_reg = max.(f.marginal,ε)

    n_marginal_grid = length(ds.marginal_grid)

    # Let us get linear functional
    post_stats = posterior_stats(ds, target_f, target_x)
    L = [z[1] for z in post_stats]

    A = ds.marginal_map;

    π1 = Convex.Variable(n_priors)
    π2 = Convex.Variable(n_priors)
    t = Convex.Variable(1)

    f1 = Convex.Variable(n_marginal_grid)
    f2 = Convex.Variable(n_marginal_grid)

    constr = [sum(π1)== 1, π1 >=0, sum(π2) == 1, π2 >= 0,
          dot(L,π1-π2) == t,
          A*π1 == f1, A*π2 == f2]

    if (C < Inf)
         push!(constr, abs(f1 - f_marginal) <= C*f_marginal_reg)
         push!(constr, abs(f2 - f_marginal) <= C*f_marginal_reg)
    end

    init_prob = maximize(t, constr)
    solve!(init_prob, GurobiSolver(NumericFocus=0, OutputFlag=0, OptimalityTol=1e-4))

    K_squared_init = norm(sqrt(m)*(f1.value-f2.value)./sqrt.(f_marginal_reg), 2)^2
    t_init = t.value[1]

    init_obj = t_init^2/(K_squared_init+4)

    main_prob = minimize(sumsquares(dot(/)(f1-f2, sqrt.(f_marginal_reg))), constr);

    Δt = t_init/10

    fix!(t, t_init - Δt)
    solve!(main_prob, GurobiSolver(NumericFocus=0, OutputFlag=0, OptimalityTol=1e-4))

    prev_obj = init_obj
    next_obj = t.value[1]^2/(4+main_prob.optval*m)

    decr = false #stop once changes very little and we have already seen change in monotonicity

    #all_obj = [prev_obj]
    #ts = [t_init]
    for k=1:max_iter
        if (decr && (abs(next_obj-prev_obj) <= prev_obj *rel_tol))
            break
        end

       if (prev_obj > next_obj)
           Δt = -Δt/2
           decr=true
       end

       if t.value[1] - Δt < 0
            Δt = Δt/10
            continue
        end
        t_next =  t.value[1] - Δt


        fix!(t, t_next)
        solve!(main_prob, GurobiSolver(NumericFocus=0, OutputFlag=0, OptimalityTol=1e-4),warmstart=true)

        prev_obj = next_obj
        next_obj = t.value[1]^2/(4+main_prob.optval*m)
        #push!(all_obj, next_obj)
        #push!(ts, t_next)
    end

    fg1 = vec(f1.value)
    fg2 = vec(f2.value)
    K_squared = main_prob.optval*m
    tg = t.value[1]

    L_g1 = dot(L, vec(π1.value))
    L_g2 = dot(L, vec(π2.value))
    L_g0 = (L_g1 + L_g2)/2

    Q = vec(m/(4+K_squared).*(fg1.-fg2)./f_marginal_reg .* tg)
    # ok the other bias term which I thought was going to 0...
    Qo = - 0.5*tg*sum( (fg1.^2 .- fg2.^2)./f_marginal_reg)
    Qo = Qo/(4 + K_squared)*m

    Qo = L_g0 + Qo

    max_bias = 2*t.value[1]/(4+K_squared)
    sd = t.value[1]*sqrt(K_squared)/(4+K_squared)
    #(max_bias, sd, max_bias^2+sd^2, next_obj)


    Q_c = BinnedCalibrator(ds.marginal_grid, Q, Qo)

    π1 = vec(π1.value)
    π2 = vec(π2.value)


    # Let us recalculate the max bias to be on the safe side

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
                      target_f,
                      target_x,
                      ε,
                      C)

    if bias_check
        opt1 = check_bias(ma)
        opt2 = check_bias(ma; maximization= false)
        max_bias = max(abs(opt1), abs(opt2))
        ma.max_bias = max_bias
    end

    ma

end


function check_bias(ma::MinimaxCalibrator; maximization=true)
    ds = ma.ds
    f = ma.f
    C = ma.C
    ε = ma.ε_reg
    target_f = ma.target_f
    target_x = ma.target_x
    Q = ma.Q

    n_priors = length(ds.priors)

    f_marginal = f.marginal
    f_marginal_reg = max.(f.marginal,ε)

    n_marginal_grid = length(ds.marginal_grid)

    A = ds.marginal_map;
    post_stats = posterior_stats(ds, target_f, target_x)
    L = [z[1] for z in post_stats]

    π3 = Convex.Variable(n_priors)
    f3 = Convex.Variable(n_marginal_grid)

    constr_max_bias = [sum(π3)== 1, π3 >=0, A*π3 == f3]

    if (C<Inf)
        push!( constr_max_bias, abs(f3 - f_marginal) <= C*f_marginal_reg)
    end

    if maximization
        max_bias_prob = maximize( Q.Qo + dot(Q.Q,f3)-dot(L,π3), constr_max_bias)
    else
        max_bias_prob = minimize( Q.Qo + dot(Q.Q,f3)-dot(L,π3), constr_max_bias)
    end

    solve!(max_bias_prob, GurobiSolver(NumericFocus=0, OutputFlag=0, OptimalityTol=1e-6))
    opt_bias = max_bias_prob.optval
    Convex.clearmemory()

    opt_bias
end
