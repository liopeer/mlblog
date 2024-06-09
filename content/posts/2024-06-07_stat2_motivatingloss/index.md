+++
title = 'Statistical Inference 2 - Motivating Differentiable Loss Functions'
date = 2024-06-06T19:37:42+02:00
draft = false
+++
{{< notice warning >}}
Incomplete!
{{< /notice >}}
In the [previous post]({{< relref "stat1_mlemap" >}}), we formulated the optimization problem, but did not talk about actually solving it. This second post introduces a case where the solution can be found analytically and shows the motivation behind the ubiquitous loss functions used in machine learning.

## Negative Log-Likelihood
We start from the likelihood function that was introduced in the previous post
$$
\begin{equation}
\bm{\theta}^\star = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta})
\end{equation}
$$
and since we have been dealing under the assumption of having access to independent samples, the joint likelihood can be factorized as
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} p(\bm{x}^{(1)}, \dots, \bm{x}^{(N)}|\bm{\theta}) = \prod_{i=1}^{N} p(\bm{x}^{(i)}|\bm{\theta}).
\end{equation}
$$
The logarithm is a montonically increasing function, therefore applying the $\log$ has no influence on the maximization operation and we can turn the product into a sum of likelihood terms for every observed sample
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} \log \left( \prod_{i=1}^{N} p(\bm{x}^{(i)}|\bm{\theta}) \right) = \argmax_{\bm{\theta}} \sum_{i=1}^{N} \log p(\bm{x}^{(i)}|\bm{\theta}).
\end{equation}
$$
It is convention to do minimization instead of maximization for most optimization problems, which can be done by inverting the sign, leading to the well known formulation of minimizing the *negative log-likelihood* (NLL).
$$
\begin{equation}
    \bm{\theta}^\star = \argmin_{\bm{\theta}} - \sum_{i=1}^{N} \log p(\bm{x}^{(i)}|\bm{\theta})
\end{equation}
$$
For very simple distributions, it is possible to derive a closed-form solution to the optimization problem by differentiating and finding stationary points, but usually optimization requires iterative optimization schemes with an appropriate differentiable loss function. 

## Loss Functions
For reasons of simplicity we will consider two cases from *supervised learning*
1. binary classification
2. regression

and w.l.o.g. we assume that $\bm{\theta}$ parameterizes a neural network $NN_\theta$ and that all our data points $\bm{x}^{(i)}$ can be split into components that belong to a *label* $\bm{y}^{(i)}$ and the complementary components $\bm{z}^{(i)}$, which will again be called *samples*. The training objective now is to infer the labels from the samples by optimizing our model.

### Regression
Inserting the labels and samples yields
$$
\begin{align}
    \bm{\theta}^\star &= \argmin_{\bm{\theta}} - \sum_{i=1}^{N} \log p(\bm{x}^{(i)}|\bm{\theta}) = \argmin_{\bm{\theta}} - \sum_{i=1}^{N} \log p(\bm{y}^{(i)},\bm{z}^{(i)}|\bm{\theta})\\
    &= \argmin_{\bm{\theta}} - \sum_{i=1}^{N} \log p(\bm{y}^{(i)}|\bm{z}^{(i)},\bm{\theta}) p(\bm{z}^{(i)}|\bm{\theta}).
\end{align}
$$
The usual assumption is that the observed data is slightly perturbed by some noise, either distributed normally around the true labels $p(\bm{y}^{(i)}|\bm{z}^{(i)}) \sim \mathcal{N}(\bm{y}^{(i)}, \bm{\Sigma}^2)$ or distributed according to a Laplacean $p(\bm{y}^{(i)}|\bm{z}^{(i)}) \sim \varphi (\bm{y}^{(i)}, \bm{\Sigma}^2)$. The prior on the samples $p(\bm{z}^{(i)})$ is usually assumed to be uniform, therefore that term is constant and drops out of the optimization. We will therefore similarly model our prediction model with the assumption of noisy predictions and uniform samples.
$$
\begin{align}
\bm{\theta}^\star &= \argmin_{\bm{\theta}} - \sum_{i=1}^{N} \log p(\bm{y}^{(i)}|\bm{z}^{(i)},\bm{\theta})
\end{align}
$$
#### Gaussian Noise
$$
\begin{align}
\bm{\theta}^\star &= \argmin_{\bm{\theta}} - \sum_{i=1}^{N} \log \mathcal{N}(\bm{y}^{(i)}; NN_\theta(\bm{x}^{(i)}),\bm{\Sigma}^2)\\
&= \argmin_{\bm{\theta}} - \sum_{i=1}^{N} (\bm{y}^{(i)} - NN_\theta(\bm{x}^{(i)}))^T (\bm{y}^{(i)} - NN_\theta(\bm{x}^{(i)}))
\end{align}
$$
#### Laplacean Noise
$$
\begin{align}
\bm{\theta}^\star &= \argmin_{\bm{\theta}} - \sum_{i=1}^{N} \log \varphi(\bm{y}^{(i)}; NN_\theta(\bm{x}^{(i)}),\bm{\Sigma}^2)\\
&= \argmin_{\bm{\theta}} - \sum_{i=1}^{N} |\bm{y}^{(i)} - NN_\theta(\bm{x}^{(i)})|
\end{align}
$$
### Classification

For classifications tasks that assume a categorical prior on the targets *cross-entropy loss* is used, regression tasks usually rely on *mean-squared error* for Gaussian priors on the error or *mean absolute error* for Laplacean priors. In the case where $\bm{\theta}$ parameterizes a neural network, the iterative optimization is almost exclusively done by variations of stochastic gradient descent (SGD). [^2] [^3] [^4]

[^2]: F. Rosenblatt. "The perceptron: a probabilistic model for information storage and organization in the brain." in Psychological Review, vol. 65, pp. 386-408, 1958.
[^3]: Diederik P. Kingma, Jimmy Ba. "Adam: A Method for Stochastic Optimization", 2017.
[^4]: Ilya Loshchilov, Frank Hutter. "Decoupled Weight Decay Regularization", 2019.