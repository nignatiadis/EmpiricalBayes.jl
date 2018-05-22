# flexible representations of priors

# Collection of priors from -- False Discovery Rates: A new deal

ash_spiky =  MixtureModel( [Normal(0,0.25); Normal(0,0.5); Normal(0,1); Normal(0,2)] ,
                           [0.4; 0.2; 0.2; 0.2 ])

ash_nearnormal = MixtureModel([Normal(0,1); Normal(0,2)], [1/3; 2/3])

ash_flattop = MixtureModel([Normal(x,0.5) for x=-1.5:0.5:1.5])

ash_skew = MixtureModel([Normal(-2,2), Normal(-1,1.5), Normal(0,1), Normal(1,1)],
                        [1/4; 1/4; 1/3; 1/6])

ash_bignormal = Normal(0,4)

ash_bimodal = MixtureModel( [Normal(-2,1); Normal(2,1)],
                            [0.5;0.5] )
