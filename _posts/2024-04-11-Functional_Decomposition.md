---
title: 'What is Functional Decomposition?'
date: 2024-04-11
permalink: /posts/2023/04/FD/
tags:
  - Functional Decomposition
  - Anchored Decomposition
  - ICE
---

This blog post discusses the topic of Functional Decomposition, which is central to understanding
XAI techniques. More specifically, we will introduce the Anchored Decomposition, its minimality property,
and how the decomposition relates to [ICE curves](https://gablabc.github.io/posts/2023/03/PDP-ICE/).

## Functional Decomposition

Let's say you have a machine learning model $h$ that takes a $d$-dimentional vector $x$ as input
and returns a prediction $h(x)$. Each component $x_j$ of the input vector is called a *feature*.
Functional Decomposition aims at describing $h(x)$ as a sum

$$h(x) = h_{\emptyset} + \sum_{j=1}^dh_{j}(x_j) + \sum_{i< j}h_{i,j}(x_i, x_j)+\ldots$$

The term $h_{\emptyset}$ is a constant called the intercept. The sub-functions $h_{j}(x_j)$ that each depend on a
single feature $x_j$ are called main-effects. Finally, the terms $h_{i,j}(x_i, x_j)$ are called
interactions and depend on several input features simultaneously. Note that interactions can go beyond
pairs of features and can include triplets $h_{ijk}(x_i, x_j, x_k)$, quadruplets etc.

How is a function decomposition supposed to help me understand my model?
To see how, one should view the task of *understanding a model* similarly to the task of *building a Lego set*.
Building an elaborate Lego construction is slow and hard if you try to build the whole structure at once.
A better strategy is to decomposes the whole structure into smaller (more digestible) chunks.
Constructing individual chunks is doable in small amounts of time and gives a constant sense of progression.
For instance, when building a medieval castle with Lego, one can first focus the entrance bridge, the watch towers,
the individual houses, the exterior walls etc. At the very end, those pieces are all aggregated to yield the final structure.
Interpreting a Machine Learning model via Functional Decomposition is similar. Instead of investigating the model
as a whole, one decomposes it into sub-components and then aims at understanding them individually.

![fd](/images/blog-bb/lego.png)

Before introducing a Functional Decomposition techniques, let us introduce additional notation.
We let $$[d]=\{1, 2\ldots, d\}$$ be the set of all input components. Given a subset of
features $u\subseteq[d]$, let the notation $x_u$ represent the vector $x$ restricted to components
in $u$, that is $x_u = (x_i)_{i\in u}$. For example, the vector $(x_1, x_2)$ can be expressed as
$$x_{12}$$. Given these additional notations, the Functional Decomposition can be expressed

$$h(x)= \sum_{u \subseteq [d]} h_u(x_u).$$

Now, how does one compute a decomposition of $h$ while remaining agnostic to its inner structure?
This topic is discussed next.

## Anchored Decomposition

The theory presented is taken from the important work:
[On Decompositions of Multivariate Functions](https://www.maths.unsw.edu.au/sites/default/files/amr08_5_0.pdf).
To compute a Functional Decomposition, the authors first let $z$ be a *fixed* reference input. Also they
let $h(x_u, z_{-u})$ be the model evaluated at a synthetic input that is a concatenation of $x_u$ and $z_{-u}$
($-u$ is the complement of $u$). The expression $h(x_u, z_{-u})$ can be interpreted as *freezing* the values of
features $-u$ to the value $z_{-u}$. Then the intercept is

$$h_{\emptyset,z} = h(z)$$

which is the model output at the reference point. Afterward, the main effects are computed

$$h_{j,z}(x_j) = h(x_j, z_{-j}) - h(z).$$

The intuition is behind these main effects is as follows: start from the reference $z$, then perturb feature 
$j$ to take value $x_j$, and finally report the increase/decrease relative to $h(z)$.

![fd](/images/blog-bb/Anchored.png)

To construct an interaction $h_{ij, z}$ the recipe is

$$h_{ij,z}(x_{ij}) = h(x_{ij}, z_{-ij}) - h_{i,z}(x_i) - h_{j,z}(x_j) - h_{\emptyset,z}.$$

This more complicated recipes consist of starting from $z$ and perturbing features $i$ and $j$ so that they take
value $x_{i}$ and $x_{j}$. After, the model prediction at that synthetic point is compared to the sum of the
intercept and the main effects of $i$ and $j$

![fd](/images/blog-bb/Anchored_2.png)

Essentially, the interaction between two features measures their how much of their joint effect cannot be explained by their
respective main effects. A general interaction $h_{u,z}$ between more than two features is given by

$$h_{u,z}(x_u) = h(x_u, z_{-u}) - \sum_{v\subset u} h_{v,z}(x_v)$$

which is a recursive formula : an interaction between features $u$ depends on its lower order interactions $v\subset u$.

## Minimality

Functional Decompositions are not unique. For example, the function $h(x) = 2x_1$ could be
decomposed as a single main-effect $h_1(x_1)=2x_1$, but it could also be decomposed as

$$h(x) = \underbrace{x_1}_{h_1(x_1)} + \underbrace{x_2}_{h_2(x_2)} +
\underbrace{(x_1 - x_2)}_{h_{12}(x_1,x_2)}.
$$

Although valid, this decomposition is undesirable because it introduces more terms than necessary.
Mathematically speaking, this decomposition is not *Minimal*. The formal definition of *Minimality* is
too complicated for the purpose of this blog so we will stick to a higher-level one

> Minimal decompositions do not introduce interaction terms unless they are really necessary to reconstruct the model output.

Minimality of the Anchored Decomposition has been proven in
[On Decompositions of Multivariate Functions](https://www.maths.unsw.edu.au/sites/default/files/amr08_5_0.pdf)
and so Equation (7) cannot occur. Indeed, let us compute the Anchored Decomposition of
the function $h(x) = 2x_1$ using $z=0$ as the reference input. The resulting intercept is

$$h_{\emptyset,z}=h(z)=0$$

The main effect of $x_1$ is

$$h_{1,z}(x_1) = h(x_1, z_{-1}) - h_{\emptyset,z} = 2x_1 - 0=2x_1$$

and that of $x_2$ is

$$h_{2,z}(x_2) = h(x_2, z_{-2}) -h_{\emptyset,z} = 0 - 0=0.$$

The interaction between features $1$ and $2$ is

$$h_{12,z}(x_{12}) = h(x_{12}, z_{-12}) - h_{1,z}(x_1) - h_{1,z}(x_1) - h_{\emptyset,z}=2x_1 - 2x_1 - 0 - 0 = 0.$$

Importantly, the Anchored Decomposition has decomposed $h$ in a model-agnostic fashion without
introducing an interaction $h_{12,z}$, which we know is unnecessary.

## Link with ICE curves

In a previous blog post we introduced [ICE curves](https://gablabc.github.io/posts/2023/03/PDP-ICE/).

$$\text{ICE}(x_j) := h(x_j, z_{-j}) - h(z).$$

Comparing this expression with $h_{j,z}(x_j)$, we can clearly see that an ICE curve is simply the
main effects of the Anchored Component using a reference $z$ sampled from the dataset. Nonetheless,
Anchored Decompositions are more general than ICE curves since one can also compute interaction terms
$h_{u,z}(x_u)$ using Anchored Decompositions.

## Takeaways

In this blog, we have learned that

1. Functional Decomposition expresses a model as a sum of sub-functions that each depend on a subset of input features.
2. The Anchored Decomposition is a model-agnostic method for decomposing arbitrary functions.
3. Anchored Decompositions are *minimal* and so will never introduce more interaction terms than
necessary to decompose the model.
