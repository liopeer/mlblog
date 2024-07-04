+++
title = 'Diffusion Models'
author = 'Lionel Peer'
date = 2024-06-20T20:33:48+02:00
draft = false
+++
## Preliminaries
All machine learning problems, that I can think of, can be formulated as learning a mapping
$$
\begin{equation}
f_\theta: \Omega \rightarrow \Lambda
\end{equation}
$$
from the sample space $\Omega$ to the label space $\Lambda$. An example is supervised learning, where a problem could be learning $f_\theta(x)\approx p(c|x)$ – the probability of sample $x$ belonging to class $c$. When we do this with neural networks, we usually use several layers, e.g. $N$, $p(c|x)\approx f_{\theta_N}\circ\dots\circ f_{\theta_1}(x)$, which we can see this as *discretizing* the problem, i.e. solving it in $N$ steps instead of a single one.
### Residual Neural Networks as Euler Solvers
Similarly, *Residual neural networks* [^1] model a discrecte sequence of transformations, but now we add the skip connection. The output of every layer can therefore be formalized as
$$
\begin{equation}
\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t, \theta_t)\quad \text{starting with some} \quad \mathbf{h}_0
\end{equation}
$$
and we can equally assume a shared set of parameters over the multiple layers, accompanied by a conditioning on $t$:
$$
\begin{equation}
\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t, \theta, t) = \mathbf{h}_t + f_{\theta}(\mathbf{h}_t, t)
\end{equation}
$$
For those familiar with numerical methods, it is immediately clear that above equation is equal to *Euler's method* [^2] for solving ODEs (ordinary differential equations) with a step size $d=1$
$$
\begin{equation}
\mathbf{h}_{t+1} = \mathbf{h}_{t} + d\cdot f(\mathbf{h}_t, t)\quad \text{with initial condition}\quad \mathbf{h}_0
\end{equation}
$$
which discretizes and approximately solves the following ODE (ordinary differential equation) with some initial condition
$$
\begin{equation}
\frac{d}{dt}\mathbf{h}(t) = f\left(\mathbf{h}(t), t\right)\quad \text{with}\quad \mathbf{h}(0) = \mathbf{h}_0
\end{equation}
$$
since
$$
\begin{align}
\mathbf{h}_{t+1} &= \mathbf{h}_{t} + d\cdot f(\mathbf{h}_t, t)\\
\frac{\mathbf{h}_{t+1} - \mathbf{h}_{t}}{d} &= f(\mathbf{h}_t, t)\\
\frac{\mathbf{h}(t+d) - \mathbf{h}(t)}{d} &= f(\mathbf{h}(t), t)\\
\xrightarrow{d\rightarrow 0}\frac{d}{dt}\mathbf{h}(t) &= f(\mathbf{h}(t), t).
\end{align}
$$
This means that a residual neural network $f_\theta$ with shared parameters in each layer $$\mathbf{h}_{t+1} = \mathbf{h}_t + f_\theta(\mathbf{h}_t, t)$$ actually can be viewed as learning finite difference approximations of the derivative $\frac{d}{dt}\mathbf{h}(t)$ and therefore the RHS (right hand side) of the ODE. Further, after summing the output of the network with the skip term, the output at each layer approximates $\mathbf{h}(t)$, the solution of our ODE by an Euler numerical solver.

### Neural ODEs
Residual neural networks in practice do not share parameters across layers and have a finite number of layers (i.e. depth of the network). For an $L$ layer network that we use for solving an ODE on the time domain $t\in [0,1]$ that means we get a fixed discretization with timestep $d=\frac{1}{L}$. *Neural ODEs* [^3] get rid of this weakness by introducing the parameters sharing from above, allowing them to be trained on arbitrary discretizations of the time domain $t$. We also don't need to fix the discretization during training, but we can train with several different discretizations and then even use again a different one for inference!



### Runge-Kutta ODE Solvers
So far we have seen that residual neural networks solve ODEs using Euler's method [^2] on a fixed discretized grid and Neural ODEs [^3] generalize them to allow for arbitrary discretization. But this ignores the fact that Euler's method is only one member of many in the *Runge Kutta* (RK) family of ODE solvers – and actually the most primitive one. *Higher-order* RK methods accumulate significantly less error at the same discretization than Euler's method, the most famous example is RK4 [^4]:
$$
\begin{align}
\mathbf{h}_{t+1} &= \mathbf{h}_{t} + \frac{d}{6}\left(k_1 + 2 k_2 + 2 k_3 + k_4\right)\\
k_1 &= f\left(\mathbf{h}_t, t\right)\\
k_2 &= f\left(\mathbf{h}_t + d\frac{k_1}{2}, t + \frac{d}{2}\right)\\
k_3 &= f\left(\mathbf{h}_t + d\frac{k_2}{2}, t + \frac{d}{2}\right)\\
k_4 &= f\left(\mathbf{h}_t + dk_3, t + d\right)\\
\end{align}
$$
As you can see, RK4 is just a composition of several $f$, so if we have a trained $f_\theta\approx f$ – a neural ODE – we can use any of the RK solvers with it and boost the accuracy of our solution!

### Training Neural ODEs
We have seen that neural ODEs approximate the time derivative of the ODE (the RHS), making it possible to use Runge-Kutta solvers for approximating the ODE's solution. But how do we actually train a neural ODE?

We start by assuming that the initial conditions to our ODE are drawn from the distribution $\mathbf{h}_0\sim p(\mathbf{h}_0)$ over all possible intial conditions - which may be a probability mass over a set of initial conditions or a continuous probability density. We further define that we want to solve the ODE on the domain $t\in\Tau$. We then sample arbitrary timesteps $t_i\in\Tau$ (e.g. uniformly $t_i \sim \mathcal{U}(\Tau)$) and train the model to recreate a sample $t_2$ from an earlier sample $t_1$, i.e. $t_1<t_2$, with the use of an RK solver like Euler or RK4. For a discretization $d$, this would mean looping over the solver $\frac{t_2-t_1}{d}$ many times and then formulate a loss between the the predicted $\hat{\mathbf{h}}(t_2)$ and the sampled $\mathbf{h}(t_2)$. In Python-inspired pseudocode for a single sample with the Euler method this would look something like this.
```python
d # discretization
h_t1 # start sample
h_t2 # target sample
for t in np.linspace(t_1, t_2, (t_2-t_1)/d):
    h_t1 = h_t1 + f(h_t1, t)
loss = mse_loss(h_t1, h_t2)
```
Since we want to train on batches it is easier to sample $t_1$ and $t_2$ such that they are $d$ apart in which case we don't actually have to sample over the RK solver.

{{% ipynb notebook="neuralODE_julia.ipynb" %}}

[^1]: He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
[^2]: Wikipedia. Euler method. https://en.wikipedia.org/wiki/Euler_method
[^3]: Chen, Ricky TQ, et al. "Neural ordinary differential equations." Advances in neural information processing systems 31 (2018).
[^4]: Wikipedia. Runge-Kutta methods. https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods