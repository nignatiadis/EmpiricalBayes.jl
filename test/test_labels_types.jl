using EmpiricalBayes
using Test

lfsr_target =  PosteriorTarget(LFSRNumerator(2.0))
mean_target = PosteriorTarget(PosteriorMeanNumerator(2.0))

EmpiricalBayes.pretty_label(lfsr_target)
EmpiricalBayes.pretty_label(mean_target)
