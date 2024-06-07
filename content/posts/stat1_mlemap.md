+++
title = 'Statistical Inference 1 - MLE & MAP'
date = 2024-06-06T14:27:08+02:00
draft = false
+++
*Maximum Likelihood Estimation* (MLE) and *Maximum a Posteriori* (MAP) estimation are fundamental concepts in statistical inference and understanding these two is *key* to understanding the motivation behind the most frequently used loss functions like *cross-entropy loss*, *mean-squared error* (MSE or L2 loss) and *mean absolute error* (L1 loss), which will be the topic of a later post.
## Introduction
Assuming a probability distribution $p: \bm{\Omega} \rightarrow \mathbb{R},\, p(\bm{x})$, independent samples $\bm{x}^{(1)}, \dots , \bm{x}^{(N)}$ and a set of parameters $\bm{\theta}$ on the domain $\bm{\Theta}$, we would like to fit a parameterized distribution $p_{\bm{\theta}}(\bm{x})$ to the original data distribution $p(\bm{x})$, possibly even in a way, such that we can generate new samples $\bm{x}^{(N+i)}\sim p_{\bm{\theta}}(\bm{x}), i>0$ that look as if they came from $p(\bm{x})$.

For simple *probability densities* over $\bm{x}$, it might be sufficient to approximate $p(\bm{x})$ by simple parameterizations such as distributions from the exponential family[^1] with suitable parameters $\bm{\theta}^\star$. These parameterizations ensure that $p_{\bm{\theta}}(\bm{x})$ is indeed a probability density and is easy to sample from:
$$
\begin{align}
    \int_{\bm{\Omega}}p_{\bm{\theta}}(\bm{x})d\bm{x} & = 1\\
    p_{\bm{\theta}}(\bm{x}) &> 0,\quad \forall \bm{x}\in \bm{\Omega}
\end{align}
$$
For arbitrary distributions over $\bm{x}$ in high-dimensional $\bm{\Omega}$ however, one has to resort to more complicated parameterizations $p_{\bm{\theta}}$ and find appropriate $\bm{\theta}^\star$. Optimizing for such $\bm{\theta}^\star$ will be the main topic of this post.

{{< notice note >}}
For low-dimensional $\bm{x}$ and sufficient number of samples it is also possible to find a discretized approximation of $p(\bm{x})$ by binning the samples and dividing the bin counts by the total number of samples (this creates a Monte Carlo estimate of the density), which ensures property (1) and the equivalent of property (2) for *probability mass functions* and is also easy to sample from:
$$
\begin{equation}
    \sum_{\forall \bm{x} \in \bm{\Omega}} p(\bm{x}) = 1
\end{equation}
$$
{{< /notice >}}

## Statistical Estimation
In this section we will try fitting probability densities $p(\bm{x})$ to data. We want our model to be such that our samples would be very highly *likely* observed if we sampled from it. This means our distribution should assign high *likelihoods* to the observed samples and we can formulate the objective as
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} p_{\bm{\theta}}(\bm{x}) = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta}).
\end{equation}
$$
For our fixed observations $\bm{x}$ and variable $\bm{\theta}$, $p(\bm{x}|\bm{\theta})$ is a function of the parameters and not of the data and is therefore usually referred to as the *likelihood* in Bayesian statistics, as opposed to a probability density. This method of estimating the optimal parameters draws its name from this function as *maximum likelihood estimation* (MLE). Existing knowledge on the distribution over $\bm{\theta}$ can be added by optimizing over the joint distribution $p(\bm{x},\bm{\theta})$, where the factorization introduces a *prior* $p(\bm{\theta})$
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} p(\bm{x},\bm{\theta}) = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta}) p(\bm{\theta})
\end{equation}
$$
Since our *evidence* $p(\bm{x})$ is constant, we can reformulate above optimization problem under Bayes' theorem as
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta}) p(\bm{\theta}) = \argmax_{\bm{\theta}} \frac{p(\bm{x}|\bm{\theta}) p(\bm{\theta})}{p(\bm{x})} = \argmax_{\bm{\theta}} p(\bm{\theta}|\bm{x})
\end{equation}
$$
which means we are effectively maximizing the *posterior* and the maximization from Eq. (5) is therefore termed *maximum a posteriori* (MAP) estimation. While usually illustrated by utilizing optimization over the parameters of a statistical model, MLE and MAP are applicable over any joint distribution, as will later be demonstrated by introducing *latent variable models* and *inverse problems*.

{{< notice note >}}
Realise that MLE is the same as MAP when considering a uniform prior $p(\bm{\theta}) = K, \forall \theta \in \Theta$, which has no influence under the maximization:
$$
\begin{equation}
    \bm{\theta}^\star = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta})p(\bm{\theta}) = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta})\cdot K = \argmax_{\bm{\theta}} p(\bm{x}|\bm{\theta})
\end{equation}
$$
{{< /notice >}}
The [next post]({{< relref "stat2_motivatingloss" >}}) will continue from the MLE and MAP formulations and motivate the commonly used loss functions.

[^1]: B. O. Koopman. "On Distributions Admitting a Sufficient Statistic" in Transactions of the American Mathematical Society, vol. 39, no. 3, pp. 399–409, 1936.