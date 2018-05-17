struct ComteButucea end

default_bandwidth(::Type{ComteButucea}, target::MarginalDensityTarget, n) = sqrt(log(n))
default_bandwidth(::Type{ComteButucea}, target::LFSRNumerator, n) = sqrt(log(n)*2/3)

default_bandwidth(::Type{ComteButucea}, n) = sqrt(log(n))
default_bandwidth(::Type{ComteButucea}, target, n) = default_bandwidth(ComteButucea, n)

function comte_butucea(z, target::InferenceTarget, n; m_bound = default_bandwidth(ComteButucea, target, n))
   f(t) = real(exp(im*t*z)*cf(target,-t)/cf(Normal(),t))
   h = m_bound # use Penski notation
   res = hquadrature(f, -h, +h)[1]
   res/2/π
end

# probably should learn how to do the above properly via FFT?

function estimate(Xs, target::InferenceTarget, ::Type{ComteButucea}; K_grid=10_000,
    marginal_grid = collect(linspace(extrema(Xs)...,K_grid)), args...)
    n = length(Xs)
    Qc = BinnedCalibrator(marginal_grid, comte_butucea.(marginal_grid, target, n; args...))
    mean(Qc.(Xs))
end
