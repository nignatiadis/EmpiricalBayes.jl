struct BinnedMarginalDensity # this encodes the estimated density..
    marginal::Vector{Float64} # measure after truncation & binning
    marginal_grid::Vector{Float64}
    marginal_h::Float64
end

function BinnedMarginalDensity(d::NormalConvolutionProblem)
    BinnedMarginalDensity(d.marginal, d.marginal_grid, d.marginal_h)
end

function BinnedMarginalDensity(M_bd::Float64, marginal_grid, marginal_h)
    BinnedMarginalDensity(ones(marginal_grid)*M_bd, marginal_grid, marginal_h);
end
