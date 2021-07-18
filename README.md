# valuable-AB-testing

## Overview
This is a personal project of mine that aims to answer the question, "How much is an A/B test worth?" The framework is based on the [value of information](https://en.wikipedia.org/wiki/Value_of_information). This repo is also used for the analyses used in my blog post, [Quantifying the value of an A/B test](https://towardsdatascience.com/quantifying-the-value-of-an-a-b-test-821aecfd2ef), which is published in Towards Data Science on Medium. The plots used in the blog post were created using AB_testing_notebook.ipynb. Note the plots are somewhat ad hoc as they are meant to be illustrative examples of the concepts discussed in the post. 

## Prerequisites
As always, it is recommended that you use a virtual environment for this repo. After creating a virual environment, the dependencies can be installed via the requirements file:

`pip install -r requirements.txt`


## Calculating the value of a test
To calculate the value of a test, you must first specify:
1. The beta distribution parameters for the prior of variant A's conversion rate
2. The beta distribution parameters for the prior of variant B's conversion rate
3. The value of each % conversion rate, i.e. how much would you pay for a variant with a (X+1)% conversion rate over one with a X% conversion rate?

For example:
`alpha = 4
beta = 100
conversion_rate_value = 10000

A = Variant(alpha, beta, conversion_rate_value)
B = Variant(alpha, beta, conversion_rate_value)`


