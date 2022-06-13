# FAIR-ML

I work for pymetrics.com, a company that specializes in reducing algorithmic bias in the hiring process. I've done quite a bit of work on tools to monitor, mitigate and report on bias, including the open-sourced package [audit-ai](https://github.com/pymetrics/audit-ai), and pymetrics' work on [auditing machine learning algorithms for fair outcomes](https://dl.acm.org/doi/pdf/10.1145/3442188.3445928).

I'm using this repo as a collection of ideas for fair ML projects that are outside the scope (and IP) of pymetrics, and are broader than the intended use of audi-ai. All ideas are my own, and are fair game for others provided you give proper citation according to the Apache 2 license.

Also, just about every project like this is called "fair-ml", so I'm going to have to think of something more clever before this gets too big.

## Data Generator

A starting issue for many people in this space is the ability to generate *biased* data so they can test against their debiasing frameworks. I've put together a simple generator that creates multivariate normal data, segments it by groups, and separates those groups by a given effect size. The first iteration will separate groups by a given effect size on an underlying continuous scale, and then use a decision rule to categorize them for classification. Correlation with generated independent variables will remain more or less the same here. In the future, I'll add some ways to generate binomial classification data with different underlying correlation matrices to independent variables (but in most test cases this will be overkill).


## Dual Optimized Regularization

The lowest-hanging fruit in debiasing ML is to identify the features that most contribute to preference of one group over another, and then to remove those features. This is a central component of audit-ai. There are a number of methods of using this information to improve algorithms, which will not be addressed here.

An emerging method to further improve on bias removal is joint optimization -- using hyperparameter tuning to regularize for fairness as well as performance. There are some interesting complications with this approach, the most pressing being that **it is often illegal to make predictions based on protected characteristics**, specifically in hiring, but also in some applications of banking, advertising, and educational settings. Instead, I'll show how to do a semi-joint optimization using holdout data. This method effectively attempts to optimize solely for performance while being orthogonal to protected characteristics.
