+++
title = "Nonparametric Quickest Change Detection using Geometric Entropy Minimization and Diffusion Maps"
date = "2025-07-28T22:17:23-04:00"
draft = true

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

I understand that this title is quite loaded, but I promise that, by the end of this post, you will understand why they all piece together to form
an interesting algorithm for detecting changes in data streams.

We will proceed first by motivating the problem of change detection, and then we will see some of the ways in which it has been approached in the past.
Then, I will introduce the idea of geometric entropy minimization, which is a (relatively) recently proposed method for performing anomaly detection in
data.
Finally, we will build upon the Geometric Entropy Minimization method to make it less sensitive to noise. We will do this by learning a diffusion map of
the data, which we will see allows us to build norms that are particularly sensitive to changes in the underlying data distribution, while being robust
to existing noise.

## Quickest Change Detection
Detecting changes in data streams is a problem that has been studied for a long time.
This is because the problem of detecting changes in statistical properties of a system or time series is one that comes up in many different fields.
For example, in finance, we may want to detect changes in the volatility of stock prices, or in the mean return of a portfolio.
In engineering, we may want to detect bifurcations in the behaviour of a system, or changes in the performance of a machine.
The list goes on, but the common theme is that we make observations that undergo a change in distribution at some point in time, and we want to detect
that change as quickly as possible, subject to some false-alarm constraints.

### Mathematical Formulation
As Veeravalli and Banerjee[^1] put it, the quickest change detection problem has three entities:
1. a stochastic process under observation,
2. a change point at which the statistical properties of the process undergo a change, and
3. a decision maker who observes the process and aims to detect the change in the process.

We aim to avoid *false alarms*, which are decisions made by the decision maker that a change has occurred when it has not.
As such, we solve the quickest change detection problem by detecting the change point as quickly as possible, subject to a constraint on the
probability of false alarm.

#### Simple Bayesian Formulation

Consider a discrete-time stochastic process \( \{X_n\} \), where \( X_n \in \mathbb{R} \) is the observation at time \( n \).
As a prior, we assume that the change point is a random variable \( \Gamma \), which takes non-negative integer values, having distribution
\( \pi_n = \mathbb{P}\{\Gamma = n\} \).
Let \( \mathbb{P}_n \) (and correspondingly \( \mathbb{E}_n \)) be the probability measure (and expectation) when the change point is at time \( \tau = n \).
We will define an algorithm which declares at time \( \tau \) whether a change has occurred or not.

Define the average detection delay \( \text{ADD} \) as
\[ \text{ADD}(\tau) = \mathbb{E} \left[ (\tau - \Gamma)^+ \right] = \sum_{n=0}^{\infty} \pi_n \mathbb{E}_n \left[ (\tau - \Gamma)^+ \right] \]
and the probability of false alarm \( \text{PFA} \) as
\[ \text{PFA}(\tau) = \mathbb{P}(\tau < \Gamma) = \sum_{n=0}^{\tau-1} \pi_n \mathbb{P}_n(\tau < \Gamma). \]
Thus, the Bayesian quickest change detection problem is to minimize \(\text{ADD}\) subject to a constraint on \(\text{PFA}\).
We can define the class of stopping times that satisfy a constraint \(\alpha\) on \(\text{PFA}\) as
\[ \mathcal{C}_{\alpha} = \{ \tau : \text{PFA}(\tau) \leq \alpha \}. \]
Then we get Shiryaev's formulation of the quickest change detection problem:
> For a given \(\alpha\), find a stopping time \(\tau^* \in \mathcal{C}_{\alpha}\) that minimizes \(\text{ADD}(\tau))\).

For simplicity, we consider the i.i.d model for the observations, and assume that \(\Gamma\) is a geometric random variable with parameter \( \rho \in (0,1) \).
That is,
\[ \pi_n = \mathbb{P}\{\Gamma = n\} = (1-\rho)^{n-1} \rho, \quad n \geq 1. \]

Let \( X_1^n = (X_1, ..., X_n) \) be the observations up to time \( n \).
We also let
\[ p_n = \mathbb{P}(\Gamma \leq n | X_1^n) \]
be the posterior probability at time \(n\) that the change has taken place given the observation up to time \(n\).
We can use Bayes' rule to show that \( p_n \) satisfies the recursion
\[ p_{n+1} = \Phi(X_{n+1}, p_n), \]
where
\[ \Phi(X_{n+1}, p_n) = \dfrac{ \tilde{p}_n L(X_{n+1}) }{ \tilde{p}_n L(X_{n+1}) + (1 - \tilde{p}_n)}, \]
where \( \tilde{p}_n = p_n + (1 - p_n) \rho\), and
\[ L(X_{n+1}) = \dfrac{ f_1(X_{n+1}) }{ f_0(X_{n+1}) } \]
is the likelihood ratio of the observations, where \( f_0 \) and \( f_1 \) are the densities of the observations before and after the change, respectively.
We define \( p_0 = 0 \).

We choose an optimal stopping time
\[ \tau_s = \inf \{ n \geq 1 : p_n \geq A \}, \]
where \( A \in (0,1) \) is a threshold that we choose based on the desired probability of false alarm.

Now let
\[ \Lambda_n = \dfrac{p_n}{(1 - p_n)} \]
and
\[ R_{n,p} = \dfrac{ p_n }{ (1 - p_n) \rho}. \]
Note that \( \Lambda_n \) is the likelihood ratio of the hypotheses "\(H_1: \Gamma \leq n\)" and "\(H_0: \Gamma > n\)" averaged over the change point:
\[
\begin{align*}
    \Lambda_n &= \dfrac{ p_n }{(1 - p_n) } \\
    &= \dfrac{ \mathbb{P}(\Gamma \leq n | X_1^n) }{ \mathbb{P}(\Gamma > n | X_1^n) } \\
    &= \dfrac{ \sum_{k=1}^n (1 - \rho)^{k-1} \rho \prod_{i=1}^{k-1} f_0 (X_i) \prod_{i=k}^n f_1 (X_i) }{(1 - \rho)^n \prod_{i=1}^n f_0(X_i) } \\
    &= \dfrac{1}{(1 - \rho)^n} \sum_{k=1}^n (1 - \rho)^{k-1} \rho \prod_{i=k}^n L(X_i).
\end{align*}
\]
Also, \(R_{n,p}\) is just a scaled version of \(\Lambda_n\):
\[ R_{n,p} = \dfrac{1}{(1 - \rho)^n} \sum_{k=1}^n (1 - \rho)^{k-1} \prod_{i=k}^n L(X_i). \]
Notably, \(R_{n,p}\) can be computed recursively as
\[ R_{n+1,p} = \dfrac{1 + R_{n,p}}{1 - \rho} L(X_{n+1}), \quad R_{0,p} = 0. \]


Thus, the Shiryaev algorithm gives us a stopping time
\[ \tau_s = \inf \{n \geq 1 : R_{n,p} \geq \frac{a}{\rho} \}, \]
with \(a = \frac{A}{1 - A}\).
Choosing \(A\) such that \(\text{PFA}(\tau_s) \leq \alpha\) gives us a stopping time that minimizes the average detection delay subject to the false alarm constraint.
The optimal threshold \(A\) must be solved numerically, and is typically done with dynamic programming. See Veeravalli and Banerjee[^1] for more details.

Note that this Bayesian formulation, although simple, requires:
1. knowledge of the distributions before and after the change, which is often not available in practice, and
2. the assumption that the change point is a geometric random variable, which may not be true in practice.

We can relax assumption number 2 by using a minimax approach, such as CuSum.

#### CuSum Approach
The CuSum (Cumulative Sum) approach is a heuristic that is often used in practice to detect changes in data streams.
The motivation for CuSum comes from the fact that, if we let \(X\) represent the generic random variable for the i.i.d model with \(f_0 \) and \(f_1\) as the densities before and after the change, respectively, then
\[ \mathbb{E}_{\infty} [\log L(X)] = -D(f_0 || f_1) < 0, \]
and
\[ \mathbb{E}_1 [\log L(X)] = D(f_1 || f_0) > 0, \]
where \(D(f_0 || f_1)\) is the Kullback-Leibler divergence between the two distributions, i.e.,
\[ D(f_1 || f_0) = \int f_1(x) \log \left( \frac{f_1(x)}{f_0(x)} \right) dx. \]
Here, \( \mathbb{E}_{\infty} \) and \( \mathbb{E}_1 \) are the expectations when the change point never occurs, and when it occurs at time 1, respectively.
Thus, after time point \( \gamma \), the log likelihood of the observation \(X\) is more likely to be above a given threshold.

By the law of large numbers,
\[ \lim_{n \to \infty} \sum_{i=1}^n \log L(X_i) = \infty. \]
Thus, if we define \(S_n = \sum_{i=1}^n \log L(X_i)\) as the accumulated log likelihood sum, then before \(\gamma\), \(S_n\) evolves towards \(-\infty\), and after \(\gamma\), it evolves towards \(+\infty\).
Therefore, we detect the change in drift and call that the change point, when $S_n$ exceeds a threshold \(b\):
\[ \tau_C = \inf \left\{ n \geq 1 : \left( S_n - \min_{1 \leq k \leq n} S_k \right) \geq b \right\}, \]
where \(b\) is a positive threshold that we choose based on the desired probability of false alarm.

Note that
\[ S_n - \min_{1 \leq k \leq n} S_k = \max_{0 \leq k \leq n} \sum_{i=k+1}^n \log L(X_i) = \max_{1 \leq k \leq n + 1} \sum_{i=k}^n \log L(X_i). \]
Thus, we can just as well write
\[ \tau_C = \inf \{ n \geq 1 : W_n \geq b \}, \]
where
\[ W_n = \max_{1 \leq k \leq n + 1} \sum_{i=k}^n \log L(X_i). \]
Note that \(W_n\) also has a nice recursive form:
\[ W_{n+1} = (W_n + \log L(X_{n+1}))^+, \quad W_0 = 0. \]

The simplicity of the CuSum approach makes it a popular choice for change detection in practice, particularly when the distributions before and after the change are known.
This is, however, a limitation, as a priori knowledge of the pre- and post-change distributions is often not available in practice.

To address this limitation, we will now introduce the idea of geometric entropy minimization, which is a method for performing anomaly detection in data without requiring knowledge of the underlying distributions.

## Geometric Entropy Minimization
First, we consider a framework for general anomaly detection.

### Anomaly Detection Framework
Let \(f_0\) denote the nominal probability distribution of \(X_t\) before any changes.
Let \( f \) denote the probability distirbution of the data stream \(X_t\) at time \(t\).
Then we can formulate the anomaly detection problem as a binary hypothesis testing problem:
\[
\begin{align*}
H_0 &: f = f_0 \quad \forall t \\
H_1 &: f = f_0, t < \tau, \text{ and } f \neq f_0, t \geq \tau,
\end{align*}
\]
where \( \tau \) denotes the unknown change point.

#### Minimum Volume Set, and its approximation
We may declare a point to be an outlier by considering the likelihood under the nominal distribution.
That is, we can say a data point is an outlier if it lies outside the most compact set of data points under the nominal distribution, called the *minimum volume set*.
The minimum volume set of level \(\alpha\) is given by
\[ \Omega_{\alpha} = \arg \min_{\mathcal{A}} \int_{\mathcal{A}} \text{d} x \text{ s.t. } \int_{\mathcal{A}} f_0(x) \text{d} x \geq 1 - \alpha, \]
where \(x\) is the data point, and \(\mathcal{A}\) is the acceptance region for \(H_0\) in which a data point is not considered an outlier, and \(\alpha\) is the significance level.
In effect, \( \alpha\) is a constraint on the false alarm rate.

\( \Omega_{\alpha} \) is called the minimum volume set because it minimizes the Lebesgue measure (volume) in \(\mathbb{R}^d\) among the subsets of data points that satisfy the false alarm constraint \(\alpha\).
That is, it minimizes \(\mathbb{P}\{\phi = H_0 | H_1 \} \), and in so doing, maximizes the detection probability \(\mathbb{P}\{\phi = H_1 | H_1\}\).

In general, it is computationally intractable to compute the minimum volume set, \( \Omega_{\alpha}\).
Thus, there have been many approaches proposed to approximate it.
One such approach is called *Geometric Entropy Minmization* (GEM)[^2].

GEM approximates the minimum volume set using minimum coverings of subsets of points in a nominal training set.
Let \(\mathcal{X}^N\) be an \(N\)-point training set of data points sampled from the nominal distribution \(f_0\).
We form a \(k\)-nearest-neighbor graph \(G = (\bar{\mathcal{X}}^N, E)\) on the training set, with \(K\) vertices \(\bar{\mathcal{X}}^N \in \mathcal{X}^N\) and \(kK\) edges
\[ E = \{ e_{i(l)} : i = 1, ..., K; l = 1, ..., k \}, \]
where the edge length \(|e_{i(l)}|\) is the Euclidean distance between the \(i\)-th vertex and its \(l\)-th nearest neighbor.
The vertices \(\bar{\mathcal{X}}^N\) are points chosen from the training set that minimize the total weighted edge length
\[ \mathcal{L}_k \left( \bar{\mathcal{X}}^N \right) = \sum_{i=1}^K \sum_{l=1}^k |e_{i(l)}|^{\gamma}, \]
over all possible \(K\)-point subsets of \(\mathcal{X}^N\), where \(\gamma > 0\) is the weight parameter.
Hero[^2] shows that \(\bar{\mathcal{X}}^N\) converges to the minimum volume set \(\Omega_{\alpha}\) as \(N \to \infty\).

Note that the above problem has exponential complexity in \(K\), and thus is not computationally tractable for the sort of large datasets that we may encounter in practice.
For this reason, we consider a bipartite kNN graph, which leads to an approach called BP-GEM.

First, we randomly partition the training set \(\mathcal{X}^N\) into two sets \(\mathcal{X^{N_1}}\) and \(\mathcal{X^{N_2}}\), with \(N_1 + N_2 = N\).
We then find vertices \(\bar{\mathcal{X}}^{N_1} \in \mathcal{X^{N_1}}\) which minimizes the total weighted edge length
\[ \mathcal{L}_k \left( \mathcal{X}_K^{N_1}, \mathcal{X}^{N_2} \right) = \sum_{i=1}^{K} \sum_{l=k-s+1}^{k} |e_{i(l)}|^{\gamma}, \]
over all possible \(K\)-point subsets of \(\mathcal{X^{N_1}}\), where \(1 \leq s \leq k\) is some fixed number.
The positive effect of this bipartite approach is that now if we get a new data point \(X_t\), we don't need to determine new representative points \(\bar{\mathcal{X}}^{N_1 + 1}\) since we choose the neighbours from the set \(\mathcal{X^{N_2}}\) which is fixed.
Thus the total edge lengths \( \sum_{l=k-s+1}^{k} |e_{i(l)}|^{\gamma} \)
does not change, and we only need to compute the total edge length for the new data point, and then choose the \(K\) points with smallest total edge lengths from the set \(\bar{\mathcal{X}}^{N_1} \cup \{X_t\}\).

#### The Change Detection Algorithm
Now that we have a way to approximate the minimum volume set, we can use it to perform change detection.
Namely, we define a function that compares the distance of a new data point \(X_t\) to the \(K\)th point in \(\bar{\mathcal{X}}^{N_1}\), which we note has the largest total edge length in \(\bar{\mathcal{X}}^{N_1}\).
This function is given by
\[ D_t = \sum_{l=k-s+1}^{k} |e_{X_t(l)}|^{\gamma} - |e_{X_{(K)}(l)}|^{\gamma}. \]

An important connection between the discrepancy measure \(D_t\) is that as \(N_1, N_2 \to \infty\), \(D_t\) converges to the log-likelihood ratio of the hypotheses \(H_0\) and \(H_1\):
\[ D_t \to \log \left( \frac{f_0(x_{\alpha})}{f_0(X_t)} \right). \]
This means that we can use \(D_t\) similar to how we used the log-likelihood ratio in the CuSum approach.

Note that in order to compute these distances, we use the Euclidean distance.
In high dimensions, however, the Euclidean distance starts to lose its discriminative power.
At this point, we consider defining a new distance measure that is more robust to noise, and which can be learned from the data.

## Diffusion Maps
Diffusion maps[^4] allow us to learn efficient geometric representations of complicated data distributions.





[^1]: Veeravalli, V. V., & Banerjee, T. (2014). Quickest change detection. In Academic press library in signal processing (Vol. 3, pp. 209-255). Elsevier.
[^2]: Hero, A. (2006). Geometric entropy minimization (GEM) for anomaly detection and localization. Advances in neural information processing systems, 19.
[^3]: Yilmaz, Y. (2017, June). Online nonparametric anomaly detection based on geometric entropy minimization. In 2017 IEEE International Symposium on Information Theory (ISIT) (pp. 3010-3014). IEEE.
[^4]: Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and computational harmonic analysis, 21(1), 5-30.
