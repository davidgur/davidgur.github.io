+++
title = "Simple Outlier Detection in a Kalman Filter"
date = "2025-12-13T00:10:57-05:00"
#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

I've honestly spent a lot of time thinking about outliers and change detection in real-time data streams.
You can get into some pretty serious rabbit holes when thinking about this stuff.
If you're not careful, you might accidentally end up in non-parametric land
with [quickest change detection](https://arxiv.org/abs/1210.5552) or some other crazy ideas[^1].

I think something a bit more down-to-earth[^2] and maybe more applicable to our day-to-day work as scientists
and engineers is the [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) which takes sequential
measurements of some underlying process (where the measurement and process errors are Gaussian) and comes up,
basically, with increasingly precise estimates of the underlying value.

I want to assume some familiarity with the Kalman filter in this post, otherwise I'll never actually finish
writing this thing.
What I aim to do here is to give you a tiny tool in your toolbox that you can apply when building a Kalman
filter in your own work.

The assumption that the measurement and process errors are jointly Gaussian[^3] is sometimes incorrect and
it is beyond your control to actually make the correction.
One instance is if the sensor you're using has some sort of failure mode.
For instance, if your sensor is some sort of positioning system, it might actually have several possible sources
of information from which it can determine your position, for example:
1. a GNSS system, like GPS, or
2. mulitlateration with the help of nearby cell towers, or
3. WiFi fingerprinting.

If you are in a place where GPS reception might be bad, which in large cities happens quite often, your phone might
be estimating your position based on other factors, like your proximity to known cell towers.
The errors here can get huge, and these failure modes might result in huge jumps.
I'm sure you can imagine some other applications where your sensor might have failure modes.
A thermometer that's been reading 26 degrees celsius for hours might accidentally jump up to 255 degrees celsius for
a second.
Even if the device combusted, I think that your first intuition as an engineer would be to reject that measurement.

Fortunately, the Kalman filter gives us the tools necessary to perform an uncertainty calculation on incoming measurements.

Suppose we have a new measurement at time \(t\), call it \(z_t\).
The *innovation* in measurement \(z_t\) is given by
\[ y_t = z_t - H x_{t|t-1}. \]
We say innovation because the idea is that \(H x_{t|t-1}\) is supposed to contain all the information that we have about the system
up to time \(t\).
You can almost think about it as \(y\) being orthogonal to the vector space of all known information, \(H x_{t|t-1}\).

Now, in the normal case, if you made the right sort of assumptions about your sensor, you will note that \(y\) is actually supposed to have
a zero mean Gaussian distribution with some covariance, \(S\).
We can actually calculate \(S\).
\[
    \begin{align*} S &= \text{Cov}(y) \\
                     &= \text{Cov}(z) - \text{Cov}(H x_{t|t-1}) \\
                     &= R + H \Sigma_{t|t-1} H^T,
    \end{align*}
\]
where \(\Sigma_{t|t-1}\) is the covariance of the estimate at time \(t\).

Now since we have a point, \(y\), and the covariance of the innovation, \(S\), we can find the [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance).
Essentially, the distance between the point and the distribution.
The distance is given by
\[ d^2 = y^T S^{-1} y. \]
What's nice about the Mahalonobis distance is that \(d^2\) follows a \(\chi_n^2\) distribution,
where \(n\) is the dimension of \(y\).
This proof falls out quite nicely when you try to take a multivariate approach to proving that the sum of squared residuals inversely scaled by the
variance also follows a \(\chi^2\) distribution.

What this means is that you can design a gate for highly unlikely measurements.
For instance if we let \(D^2 \sim \chi^2_2\), then we have \(\mathbb{P}(D^2 > 9.21) = 0.01\).
This means that we can reject cases where \(d^2 > 9.21\), because, given all that we know about the system at this moment in time,
there is less than a \(1\%\) probability of that measurement occuring.

In code, this might look something like the following:
```python
def update(self, z):
    y = z - self.H @ self.x
    S = self.H @ self.P @ self.H.T + R
    K = self.P @ self.H.T @ np.linalg.inv(S)

    # Mahalanobis gate
    maha2 = y.T @ np.linalg.inv(S) @ y
    threshold = 9.210 # chi-square 2 DOF, 99% confidence

    if maha2 <= threshold:
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
```

This code will flat-out ignore measurements that don't make it past the gate.
Depending on the expected experience, you might also want this to trigger an increase in either the process
error covariance or the sensor error covariance, so that you can demonstrate more uncertainty in your final estimate.

I think it's also worth noting that you shouldn't use this as a crutch in critical situations,
but I'd hope that if you're reading this and designin mission-critical systems, you don't take advice from a blog in any case.

In my experience, this method works well for cases where you want to deliver smooth measurements while
having faulty sensors that are outside of your control.




[^1]: These ideas are not actually crazy, they're very cool and something I want to work with more.
[^2]: The joke here is the [connection of the Kalman filter to the space program](https://ntrs.nasa.gov/api/citations/19860003843/downloads/19860003843.pdf).
[^3]: The assumption isn't really about the errors being Gaussian, it's more about having conditions necessary so that the minimum mean square error (MMSE) estimator is linear and is indeed the optimal estimator. This just happens to be the case when the process noise and the measurement noise is jointly Gaussian.
