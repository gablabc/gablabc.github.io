---
title: "Fooling SHAP with Stealthily Biased Sampling "
collection: publications
permalink: /publication/fool_SHAP
excerpt: 'Manipulating Shapley values by cherry-picking the reference samples'
date: 2023-05-01
venue: 'ICLR'
citation: 'Laberge, G., Aïvodji, U., Hara, S., Marchand, M., & Khomh, F. (2023, May). 
Fooling SHAP with Stealthily Biased Sampling. In The Eleventh International Conference on Learning Representations.'
---

This paper demonstrates the possibility of **cherry picking** the data samples provided to SHAP in
order to change the global feature importance. The specific use-case presented concerns a model
audit where a company has to convince an auditor that the disparities in model outcome among
protected subgroups are caused by *meritocratic* features.

![foolshap](/images/papers/attack_logo.png)

[Paper](https://openreview.net/pdf?id=J4mJjotSauh)

[Github](https://github.com/gablabc/Fool_SHAP)