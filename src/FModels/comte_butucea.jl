struct ComteButucea
    Q::BinnedCalibrator
    h::Float64
    target::InferenceTarget
end

default_bandwidth(::Type{ComteButucea}, target::MarginalDensityTarget, n) = sqrt(log(n))
default_bandwidth(::Type{ComteButucea}, target::LFSRNumerator, n) = sqrt(log(n)*2/3)

default_bandwidth(::Type{ComteButucea}, n) = sqrt(log(n))
default_bandwidth(::Type{ComteButucea}, target, n) = default_bandwidth(ComteButucea, n)

function comte_butucea(z, h, target::InferenceTarget)
   f(t) = real(exp(im*t*z)*cf(target,-t)/cf(Normal(),t))
   res = hquadrature(f, -h, +h)[1]
   res/2/π
end

function ComteButucea(target::InferenceTarget, m::Int64, marginal_grid;
    h= default_bandwidth(ComteButucea, target, m))
    Qc = BinnedCalibrator(marginal_grid, comte_butucea.(marginal_grid, h, target))
    ComteButucea(Qc, h, target)
end

function estimate(Xs,cb::ComteButucea)
    mean(cb.Q.(Xs))
end

function estimate(Xs, ::Type{ComteButucea}, target::InferenceTarget, marginal_grid)
    cb = ComteButucea(target, length(Xs), marginal_grid)
    estimate(Xs, cb)
end
# probably should learn how to do the above properly via FFT?

function estimate(Xs, target::InferenceTarget, ::Type{ComteButucea}; K_grid=10_000,
    marginal_grid = collect(linspace(extrema(Xs)...,K_grid)), args...)
    n = length(Xs)
    Qc = BinnedCalibrator(marginal_grid, comte_butucea.(marginal_grid, target, n; args...))
    mean(Qc.(Xs))
end
