---
title: "Learning Hybrid Interpretable Models: Theory, Taxonomy, and Methods"
collection: publications
category: manuscripts
permalink: /publication/Hybrid
excerpt: 'Explore the Transparency-Accuracy tradeoff'
date: 2024-09-05
venue: 'TMLR'
citation: 'Ferry, J., Laberge, G., Aivodji, U. (2024). Learning Hybrid Interpretable Models: Theory, Taxonomy, and Methods. Transactions on Machine Learning Research, 2835-8856.'
---

A Hybrid Interpretable Model (HIM) involves the cooperation of an interpretable model and a complex black box. At inference, any input of the HIM is assigned to either its interpretable 
or complex component based on a gating mechanism. The ratio of data samples sent to the interpretable component is referred to as the transparency. 
Despite their high potential, HIMs remain under-studied in the interpretability/explainability literature. In this paper, we remedy this fact by 
presenting a thorough investigation of such models from three perspectives: Theory, Taxonomy, and Methods. 
First, we highlight the potential generalization benefits of sending samples to an interpretable component by deriving a Probably-Approximately-Correct (PAC) generalization 
bound. Secondly, we provide a general taxonomy for the different ways of training such models: the Post-Black-Box and Pre-Black-Box paradigms. These approaches differ in the 
order in which the interpretable and complex components are trained. Thirdly, we implement the two paradigms in a single method: HybridCORELS, which extends the CORELS 
algorithm to Hybrid Interpretable Modeling. By leveraging CORELS, HybridCORELS provides a certificate of optimality of its interpretable component and precise control 
over transparency. We finally show empirically that HybridCORELS is competitive with existing approaches and performs just as well as a standalone black box (or even better) 
while being partly transparent.

![hybrid](/images/papers/Hybrid.png)

[Paper](https://openreview.net/forum?id=XzaSGIStXP)

[Github](https://github.com/ferryjul/hybridcorels)
