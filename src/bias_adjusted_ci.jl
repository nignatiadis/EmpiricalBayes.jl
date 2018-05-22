function get_plus_minus(maxbias, se, conf=0.9)
    rel_bias = maxbias/se
    zz = fzero( z-> cdf(Normal(), rel_bias-z) + cdf(Normal(), -rel_bias-z) + conf -1,
        0, rel_bias - quantile(Normal(),(1-conf)/3))
    zz*se
end
