struct BinnedMarginalDensity # this encodes the estimated density..
    marginal::Vector{Float64} # measure after truncation & binning
    marginal_grid::Vector{Float64}
    marginal_h::Float64
end

function BinnedMarginalDensity(d::NormalConvolutionProblem)
    BinnedMarginalDensity(d.marginal, d.marginal_grid, d.marginal_h)
end
