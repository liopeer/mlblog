+++
title = 'Statistical Inference 1 - MLE & MAP'
date = 2024-06-06T14:27:08+02:00
draft = false
+++
## Introduction
*Maximum Likelihood Estimation* (MLE) and *Maximum a Posteriori* (MAP) estimation are fundamental concepts in statistical inference and understanding these two is **key** to understanding the motivation behind the most frequently used loss functions like *cross-entropy loss*, *mean-squared error* (MSE or L2 loss) and *mean absolute error* (L1 loss).

Assuming a probability distribution $p: \bm{\Omega} \rightarrow \mathbb{R},\, p(\bm{x})$, independent samples $\bm{x}^{(1)}, \dots , \bm{x}^{(N)}$ and a set of parameters $\bm{\theta}$ on the domain $\bm{\Theta}$, *generative modeling* is occupied with fitting a parameterized distribution $p_{\bm{\theta}}(\bm{x})$ to the original data distribution $p(\bm{x})$, such that we can generate new samples $\bm{x}^{(N+i)}\sim p_{\bm{\theta}}(\bm{x}), i>0$ that look as if they came from $p(\bm{x})$.

For simple *probability densities* over $\bm{x}$, it might be sufficient to approximate $p(\bm{x})$ by simple parameterizations such as distributions from the exponential family[^1] with suitable parameters $\bm{\theta}^\star$. These parameterizations ensure that $p_{\bm{\theta}}(\bm{x})$ is indeed a probability density and is easy to sample from:
$$
\begin{align}
    \int_{\bm{\Omega}}p_{\bm{\theta}}(\bm{x})d\bm{x} & = 1\\
    p_{\bm{\theta}}(\bm{x}) &> 0,\quad \forall \bm{x}\in \bm{\Omega}
\end{align}
$$
For low-dimensional $\bm{x}$ it is also possible to find a discretized approximation of $p(\bm{x})$ by binning the samples and dividing the bin counts by the total number of samples, which ensures property (1) and the equivalent of property (2) for *probability mass functions* and is also easy to sample from:
$$
\begin{equation}
    \sum_{\forall \bm{x} \in \bm{\Omega}} p(\bm{x}) = 1
\end{equation}
$$

For arbitrary distributions over $\bm{x}$ in high-dimensional $\bm{\Omega}$ however, one has to resort to more complicated parameterizations $p_{\bm{\theta}}$.

## Statistical Estimation
For this section, we will relax the constraint of fitting probability densities $p(\bm{x})$ to data, but will instead consider fitting generic functions $f(\bm{x})$ to observed data points that approximate the density up to a constant factor $K$. Since we ultimately want our model to assign high probabilities to the observed samples, we can formulate the objective as
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} p_{\bm{\theta}}(\bm{x}) = \argmax_{\bm{\theta}} \frac{1}{K} f_{\bm{\bm{\theta}}}(\bm{x}) = \argmax_{\bm{\theta}} f(\bm{x}|\bm{\theta}).
\end{equation}
$$
For fixed observations $\bm{x}$ and variable $\bm{\theta}$, $p(\bm{x}|\bm{\theta})$ is a function of the parameters and not of the data and is therefore usually referred to as the *likelihood* in Bayesian statistics, as opposed to a probability density. This method of estimating the optimal parameters draws its name from this function as *maximum likelihood estimation* (MLE). Existing knowledge on the distribution over $\bm{\theta}$ can be added by optimizing over the joint distribution $p(\bm{x},\bm{\theta})$, where the factorization introduces a *prior* $p(\bm{\theta})$
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta}) = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta}) p(\bm{\theta})
\end{equation}
$$
Since our *evidence* $p(\bm{x})$ is constant, we can reformulate above optimization problem under Bayes' theorem as
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta}) p(\bm{\theta}) = \argmax_{\bm{\theta}} \frac{p(\bm{x}|\bm{\theta}) p(\bm{\theta})}{p(\bm{x})} = \argmax_{\bm{\theta}} p(\bm{\theta}|\bm{x})
\end{equation}
$$
which means we are effectively maximizing the *posterior* and this maximization is therefore termed *maximum a posteriori* (MAP) estimation. While usually illustrated by utilizing optimization over the parameters of a statistical model, MLE and MAP are applicable over any joint distribution, as will later be demonstrated by introducing *latent variable models* and *inverse problems*.

Under the assumption of independent samples from the previous section~\ref{sec:genmodintro}, the joint likelihood can be factorized as
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} p(\bm{x}^{(1)}, \dots, \bm{x}^{(N)}|\bm{\theta}) = \prod_{i=1}^{N} p(\bm{x}^{(i)}|\bm{\theta})
\end{equation}
$$
and since maximizing the $\log$ of a function has no influence on the maximization operation we can turn the product into a sum, which is usually easier to use as an optimization objective
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
For very simple distributions, it is possible to derive a closed-form solution to the optimization problem by differentiating and finding stationary points, but usually optimization requires iterative optimization schemes with an appropriate differentiable loss function. For classifications tasks that assume a categorical prior on the targets *cross-entropy loss* is used, regression tasks usually rely on *mean-squared error* for Gaussian priors on the error or *mean absolute error* for Laplacean priors. In the case where $\bm{\theta}$ parameterizes a neural network, the iterative optimization is almost exclusively done by variations of stochastic gradient descent (SGD).[^2][^3][^4]

[^1]: B. O. Koopman. "On Distributions Admitting a Sufficient Statistic" in Transactions of the American Mathematical Society, vol. 39, no. 3, pp. 399–409, 1936.
[^2]: F. Rosenblatt. "The perceptron: a probabilistic model for information storage and organization in the brain." in Psychological Review, vol. 65, pp. 386-408, 1958.
[^3]: Diederik P. Kingma, Jimmy Ba. "Adam: A Method for Stochastic Optimization", 2017.
[^4]: Ilya Loshchilov, Frank Hutter. "Decoupled Weight Decay Regularization", 2019.