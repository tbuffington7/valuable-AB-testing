# valuable-AB-testing

## February 2024 update
I have gained a lot of experience with A/B testing in the last few years since doing this project. Although I think the VoI framework is sound, there are a few practical issues that I would like to address in the future.
- This repo places priors on both variants _separately_, which I believe leads to the [Bayesian imposter](https://www.geteppo.com/blog/beware-of-the-bayesian-imposter).
- It would be better to place priors on the relative lift.
- Because the priors are not optimally specified, the repo recommends sample sizes that are laughably small. When I have bandwidth, I plan to update this repo so it uses more realistic priors, which should generate more realistic sample size recommendations.

## Overview
This is a personal project of mine that aims to answer the question, "How much is an A/B test worth?" The framework is based on the [value of information](https://en.wikipedia.org/wiki/Value_of_information).

Currently the repo allows one to compute the value of an A/B test for a conversion rate. This lends itself to a Beta/Binomial model. 

## Prerequisites
As always, it is recommended that you use a virtual environment for this repo. After creating a virual environment, the dependencies can be installed via the requirements file:

`pip install -r requirements.txt`


## Calculating the value of a test
To calculate the value of a test, you must first specify:
1. The beta distribution parameters for the prior of variant A's conversion rate
2. The beta distribution parameters for the prior of variant B's conversion rate
3. The value of each % conversion rate, i.e. how much would you pay for a variant with a (X+1)% conversion rate over one with a X% conversion rate?

For example:
```
alpha = 4
beta = 100
conversion_rate_value = 10000 # USD/% conversion rate

A = Variant(alpha, beta, conversion_rate_value)
B = Variant(alpha, beta, conversion_rate_value)
```

Note that although the two variants have the same prior in this example, you can specify different priors for the two variants. 

After instantating the `Variant` objects, you can compute the value of an A/B test with, for example, 100 participants (for each variant) like this:
`voi = calc_voi(A, B, test_sample_size=100)`



