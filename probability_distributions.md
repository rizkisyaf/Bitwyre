# Probability Distributions: Mathematical Derivations

This document provides the mathematical derivations of probability density functions (PDFs) and cumulative distribution functions (CDFs) for three common probability distributions:

1. Gaussian (Normal) Distribution
2. Poisson Distribution
3. Uniform Distribution

## 1. Gaussian (Normal) Distribution

### Probability Density Function (PDF)

The Gaussian distribution is characterized by its bell-shaped curve and is defined by two parameters: the mean (μ) and the standard deviation (σ).

The PDF of a Gaussian distribution is given by:

$$f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

#### Derivation

The Gaussian distribution can be derived from the principle of maximum entropy. If we know only the mean and variance of a random variable, the distribution that maximizes entropy (i.e., assumes the least additional information) is the Gaussian distribution.

Starting with the entropy functional:

$$S[f] = -\int_{-\infty}^{\infty} f(x) \ln f(x) dx$$

And applying the constraints:

1. $\int_{-\infty}^{\infty} f(x) dx = 1$ (normalization)
2. $\int_{-\infty}^{\infty} x f(x) dx = \mu$ (mean)
3. $\int_{-\infty}^{\infty} (x-\mu)^2 f(x) dx = \sigma^2$ (variance)

Using the method of Lagrange multipliers, we can maximize $S[f]$ subject to these constraints, which leads to the Gaussian PDF.

### Cumulative Distribution Function (CDF)

The CDF of a Gaussian distribution is:

$$F(x; \mu, \sigma) = \int_{-\infty}^{x} f(t; \mu, \sigma) dt = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$$

where $\text{erf}(z)$ is the error function defined as:

$$\text{erf}(z) = \frac{2}{\sqrt{\pi}}\int_{0}^{z} e^{-t^2} dt$$

#### Derivation

To derive the CDF, we integrate the PDF from $-\infty$ to $x$:

$$F(x; \mu, \sigma) = \int_{-\infty}^{x} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{t-\mu}{\sigma}\right)^2} dt$$

Using the substitution $u = \frac{t-\mu}{\sigma\sqrt{2}}$, we get:

$$F(x; \mu, \sigma) = \int_{-\infty}^{\frac{x-\mu}{\sigma\sqrt{2}}} \frac{1}{\sqrt{\pi}} e^{-u^2} d(\sigma\sqrt{2}u + \mu)$$

$$F(x; \mu, \sigma) = \int_{-\infty}^{\frac{x-\mu}{\sigma\sqrt{2}}} \frac{\sigma\sqrt{2}}{\sqrt{\pi}} e^{-u^2} du$$

$$F(x; \mu, \sigma) = \frac{1}{2} + \frac{1}{\sqrt{\pi}}\int_{0}^{\frac{x-\mu}{\sigma\sqrt{2}}} e^{-u^2} du$$

$$F(x; \mu, \sigma) = \frac{1}{2} + \frac{1}{2}\text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)$$

$$F(x; \mu, \sigma) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$$

## 2. Poisson Distribution

### Probability Mass Function (PMF)

The Poisson distribution models the number of events occurring in a fixed time interval, given the average rate of occurrence.

The PMF of a Poisson distribution is:

$$P(X = k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$

where:
- $\lambda$ (lambda) is the average rate of events per interval
- $k$ is the number of events (non-negative integer)
- $e$ is Euler's number (≈ 2.71828)
- $k!$ is the factorial of $k$

#### Derivation

The Poisson distribution can be derived as a limit of the binomial distribution when the number of trials $n$ approaches infinity and the probability of success $p$ approaches zero, while the product $np = \lambda$ remains constant.

Starting with the binomial PMF:

$$P(X = k; n, p) = \binom{n}{k} p^k (1-p)^{n-k}$$

As $n \to \infty$ and $p \to 0$ with $np = \lambda$:

$$\lim_{n \to \infty, p \to 0, np = \lambda} \binom{n}{k} p^k (1-p)^{n-k}$$

$$= \lim_{n \to \infty} \frac{n!}{k!(n-k)!} \left(\frac{\lambda}{n}\right)^k \left(1-\frac{\lambda}{n}\right)^{n-k}$$

$$= \lim_{n \to \infty} \frac{n(n-1)\cdots(n-k+1)}{k!} \left(\frac{\lambda}{n}\right)^k \left(1-\frac{\lambda}{n}\right)^{n} \left(1-\frac{\lambda}{n}\right)^{-k}$$

As $n \to \infty$:
- $\frac{n(n-1)\cdots(n-k+1)}{n^k} \to 1$
- $\left(1-\frac{\lambda}{n}\right)^{n} \to e^{-\lambda}$
- $\left(1-\frac{\lambda}{n}\right)^{-k} \to 1$

Therefore:

$$P(X = k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$

### Cumulative Distribution Function (CDF)

The CDF of a Poisson distribution is:

$$F(k; \lambda) = \sum_{i=0}^{\lfloor k \rfloor} \frac{\lambda^i e^{-\lambda}}{i!}$$

where $\lfloor k \rfloor$ is the floor function (largest integer not greater than $k$).

#### Derivation

Since the Poisson distribution is discrete, the CDF is simply the sum of the PMF values from 0 to $k$:

$$F(k; \lambda) = \sum_{i=0}^{\lfloor k \rfloor} P(X = i; \lambda) = \sum_{i=0}^{\lfloor k \rfloor} \frac{\lambda^i e^{-\lambda}}{i!}$$

For non-integer $k$, we use the floor function to sum up to the largest integer not exceeding $k$.

## 3. Uniform Distribution

### Probability Density Function (PDF)

The uniform distribution represents equal probability over a range $[a, b]$.

The PDF of a uniform distribution is:

$$f(x; a, b) = \begin{cases}
\frac{1}{b-a} & \text{for } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}$$

#### Derivation

The uniform distribution is derived from the principle that the probability is equally distributed over the interval $[a, b]$.

Since the total probability must be 1, and the probability density is constant over the interval, we have:

$$\int_{a}^{b} f(x) dx = 1$$

If $f(x) = c$ (constant) for $a \leq x \leq b$, then:

$$\int_{a}^{b} c \, dx = c(b-a) = 1$$

Therefore:

$$c = \frac{1}{b-a}$$

And the PDF is:

$$f(x; a, b) = \begin{cases}
\frac{1}{b-a} & \text{for } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}$$

### Cumulative Distribution Function (CDF)

The CDF of a uniform distribution is:

$$F(x; a, b) = \begin{cases}
0 & \text{for } x < a \\
\frac{x-a}{b-a} & \text{for } a \leq x \leq b \\
1 & \text{for } x > b
\end{cases}$$

#### Derivation

To derive the CDF, we integrate the PDF from $-\infty$ to $x$:

For $x < a$:
$$F(x; a, b) = \int_{-\infty}^{x} f(t; a, b) dt = \int_{-\infty}^{x} 0 \, dt = 0$$

For $a \leq x \leq b$:
$$F(x; a, b) = \int_{-\infty}^{x} f(t; a, b) dt = \int_{-\infty}^{a} 0 \, dt + \int_{a}^{x} \frac{1}{b-a} dt = 0 + \frac{x-a}{b-a} = \frac{x-a}{b-a}$$

For $x > b$:
$$F(x; a, b) = \int_{-\infty}^{x} f(t; a, b) dt = \int_{-\infty}^{a} 0 \, dt + \int_{a}^{b} \frac{1}{b-a} dt + \int_{b}^{x} 0 \, dt = 0 + 1 + 0 = 1$$

Therefore:

$$F(x; a, b) = \begin{cases}
0 & \text{for } x < a \\
\frac{x-a}{b-a} & \text{for } a \leq x \leq b \\
1 & \text{for } x > b
\end{cases}$$ 