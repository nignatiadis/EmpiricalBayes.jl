RecipesBase.@recipe function f(ma::LinearEstimator)
    RecipesBase.@series begin
        label --> pretty_label(ma) #TODO, make more generic..
        seriestype --> :line
        ma.Q.marginal_grid,  ma.Q.Q .+ ma.Q.Qo
    end
end
