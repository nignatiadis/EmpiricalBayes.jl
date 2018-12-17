function smooth_poly_max(x::Float64)
    if x<=zero(Float64)
        return(zero(Float64))
    elseif x>= one(Float64)
        return(x)
    else
        return( 2x^2 - x^3)
    end
end

function smooth_poly_max(x, ϵ, κ)
    mult_λ = (κ+1)*ϵ
    smooth_poly_max((x-ϵ)/mult_λ)*mult_λ + ϵ
end

function supersmooth(x::Float64)
   if x<=zero(Float64)
        return(zero(Float64))
   else
        return(exp(-1/x))
   end
end

function supersmooth_max(x)
    x*(supersmooth(x)/(supersmooth(x) + supersmooth(1-x)))
end

function supersmooth_max(x, ϵ, κ)
    mult_λ = (κ+1)*ϵ
    supersmooth_max((x-ϵ)/mult_λ)*mult_λ + ϵ
end
