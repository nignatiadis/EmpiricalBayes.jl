# from the classical Annals paper..

# To do this properly with KernelDensity.jl later,
# also figure out generic interface

struct BrownGreenshtein
    m::Int64
    bandwidth::Float64
end

BrownGreenshtein(m) = BrownGreenshtein(m, 1/sqrt(log(m)))


function estimate(Xs, ::Type{BrownGreenshtein}, target::PosteriorTarget{PosteriorMeanNumerator})
    bg = BrownGreenshtein(length(Xs))
    estimate(Xs, bg, target)
end


function estimate(Xs, bg::BrownGreenshtein, target::PosteriorTarget{PosteriorMeanNumerator})
    x = target.num.x
    h = bg.bandwidth
    Z = Normal(0, 1)
    @show denom = mean( pdf.(Z, (Xs.-x)./h))/h
    num = mean( (Xs.-x).*pdf.(Z, (Xs.-x)./h) )/(h^3)
    x + num/denom
end
