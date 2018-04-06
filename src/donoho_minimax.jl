struct BinnedCalibrator
    marginal_grid::Vector{Float64}
    marginal_h::Float64
    Q::Vector{Float64}
end

function BinnedCalibrator(marginal_grid, Q) 
    marginal_h = marginal_grid[2] - marginal_grid[1]
    BinnedCalibrator(marginal_grid, marginal_h, Q)
end

function (c::BinnedCalibrator)(x)
    marginal_grid = c.marginal_grid
    marginal_h = c.marginal_h
    Q = c.Q
    # TODO: Some checks that we don't violate stuff..
    # Vectorized version of this..?
    idx = searchsortedlast(marginal_grid .- marginal_h/2, x)
    idx = max(1, min(idx, length(marginal_grid)))
    Q[idx]
end