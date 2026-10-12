# # DLMDSAS01 - Advanced Statistics

# #### - Imports -

from scipy.special import comb, kv, kve
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy import stats
from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

# ## Task 1 - Basic Probabilities and Visualizations (1)

# ### Parameters

k = 78      # ξ2
p = 0.49    # ξ3

# ### PMF

def pmf(m):
    return comb(m + k - 1, k - 1) * (p**k) * ((1 - p)**m)

# ### Distribution

m_vals = []
pmf_vals = []

m = 0
peak_passed = False

while True:
    prob = pmf(m)
    
    m_vals.append(m)
    pmf_vals.append(prob)
    
    if m > 0 and pmf_vals[-1] < pmf_vals[-2]:
        peak_passed = True
    
    if peak_passed and prob < 0.005:
        break
    
    m += 1

m_vals = np.array(m_vals)
pmf_vals = np.array(pmf_vals)

# ### Expectation (mean)

mean = k * (1 - p) / p

print(f"Expectation (mean): {mean:.4f}")

# ### Median

cum = np.cumsum(pmf_vals)
median = np.searchsorted(cum, 0.5)

print(f"Median: {median}")

# ### Graphic Visualization

plt.figure()
plt.bar(m_vals, pmf_vals)

plt.axvline(mean, linestyle='--', color="springgreen", label=f"Expectation (mean) ≈ {mean:.2f}")
plt.axvline(median, linestyle=':', color="red", label=f"Median = {median}")

plt.xlabel("Number of meteorites (m)")
plt.ylabel("Probability P(M=m)")
plt.title("Negative Binomial Distribution")

plt.legend()

plt.savefig("Graphics/graphic_task1.pdf", format='pdf', dpi=300, bbox_inches='tight')

plt.show()

# ## Task 2 - Basic Probabilities and Visualizations (2)

# ### Parameters

a = 2/9    # ξ5
b = 4      # ξ6
c = 7/9    # ξ7
d = 9      # ξ8

# ### Survival function

def s(y):
    return (a * np.exp(-b * y**2) + c * np.exp(-d * y**8))

# ### Computation of Probability

probability = s(2) - s(4)
print(f"{probability:.10f}")

# ### PDF (Probability Density Function)

def f(y):
    return (2 * a * b * y * np.exp(-b * y**2) + 8 * c * d * y**7 * np.exp(-d * y**8))

# ### CDF (Cumulative Density Function)

def cdf_task2(y):
    return (1 - s(y))

# ###  Mean, Variance, and Quartiles

mean, _ = quad(lambda y: y * f(y), 0, np.inf)

second_moment, _ = quad(lambda y: y**2 * f(y), 0, np.inf)
variance = second_moment - mean**2

q25 = brentq(lambda y: cdf_task2(y) - 0.25, 0, 2)
q50 = brentq(lambda y: cdf_task2(y) - 0.5, 0, 2)
q75 = brentq(lambda y: cdf_task2(y) - 0.75, 0, 2)

print(f"Mean: {mean:.2f}")
print(f"Variance: {variance:.2f}")
print(f"First Quartile (25th percentile): {q25:.2f}")
print(f"Second Quartile (50th percentile / Median): {q50:.2f}")
print(f"Third Quartile (75th percentile): {q75:.2f}")

# ### PDF Graph

y = np.linspace(0, 2, 1000)

plt.plot(y, f(y))

plt.axvline(mean, linestyle='--', color="springgreen", label='Mean')
plt.axvline(q25, linestyle=':', color="orange", label='Q1')
plt.axvline(q50, linestyle='-.', color="red", label='Median')
plt.axvline(q75, linestyle=':', color="orange", label='Q3')

plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel("Density f(y)")
plt.title("Probability Density Function")

plt.savefig("Graphics/pdf_task2.pdf", format='pdf', dpi=300, bbox_inches='tight')

plt.show()

# ### Histogram

minutes = np.arange(0, 120)
y_minutes = minutes / 60

delta = 1/60
probs = f(y_minutes) * delta

plt.bar(minutes, probs)

plt.axvline(mean*60, linestyle='--', color="springgreen",label='Mean')
plt.axvline(q25*60, linestyle=':', color="orange", label='Q1')
plt.axvline(q50*60, linestyle='-.', color="red", label='Median')
plt.axvline(q75*60, linestyle=':', color="orange", label='Q3')

plt.legend()
plt.xlabel("Time (minutes)")
plt.ylabel("Probability")
plt.title("Probability of hearing the owl per minute")

plt.savefig("Graphics/histogram_task2.pdf", format='pdf', dpi=300, bbox_inches='tight')

plt.show()

# ## Task 3 - Transformed Random Variables

# ### Parameters

bandwidth_totals = [37, 52, 64, 18, 0]  # ξ10

# ### Maximum Likelihood Estimator

theta_hat = np.mean(bandwidth_totals) / 2

# ### Expectation of T

E_T = 2 * theta_hat

# ### Results

print("MLE theta:", theta_hat)
print("Expected T:", E_T)

# ## Task 4 - Hypothesis Test

# ### Parameters

samples = [928, 938, 950, 954, 941, 936, 1008, 937, 939, 966]   # ξ14
n = len(samples)
sigma0 = 35.2       # ξ12
alpha = 0.05

# ### Mean

mean = np.mean(samples)
print(f"Mean: {mean}")

# ### Variance

s2 = np.var(samples, ddof=1)
print(f"Variance: {s2:.2f}")

# ### Chi-square test statistic

chi_square = (n-1)*s2/(sigma0**2)
print(f"Chi-square: {chi_square:.3f}")

# #### Critical Value for Left-tailed test

critical = chi2.ppf(alpha, df=n-1)
print(f"Critical Value for Left-tailed test: {critical:.3f}")

# #### p-value

p_value = chi2.cdf(chi_square, df=n-1)
print(f"p-value: {p_value:.3f}")

# ### Type II error probability

# assume the new system has a standard deviation of 30 g

sigma1 = 30

beta = 1 - chi2.cdf(
    critical * (sigma0**2 / sigma1**2),
    df=n-1
)

print(f"Type II error probability (beta): {beta:.3f}")

# ### Power of the test

power = 1 - beta

print(f"Power (1 - beta): {power:.3f}")

# ## Task 5: Regularized Regression

# ### Parameters

data = [(-6, 6038422291.73), (-7, 37102435277.92), (6, 7987712684.26),
(-14, 155345310475368.06), (-12, 24623459385110.95), (10,
3248796791320.71), (4, 68668227.69), (-16, 833719741222972.5),
(-17, 1641343452564754), (14, 187943996770917.84), (-19,
6375474630693752), (-8, 196293759232.3), (1, 38.08), (15,
434811484112158.25), (17, 1797962615235692.2), (7,
50605917639.08), (-13, 64849485988643.69), (2, 27217.5), (9,
937162591525.51), (-1, -4.06), (16, 901265743566097.1), (3,
2564389.54), (20, 12489064946603642), (-15, 366972436510100.4),
(-18, 3361052474760982), (19, 7039505527892020)]    # ξ15

x_data, y_data = zip(*data)

x = np.array(x_data)
y = np.array(y_data)

# ### Design matrix

X = np.column_stack([x**i for i in range(13)])

alpha_ols, _, _, _ = np.linalg.lstsq(
    X, y, rcond=None
)

# ### OLS

alpha_ols, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

y_pred_ols = X @ alpha_ols

sse_ols = np.sum((y - y_pred_ols)**2)

print("\nOLS coefficients:")
print(alpha_ols)

print("\nOLS SSE:")
print(sse_ols)

# ### Ridge

lambdas = [0.0, 1.0, 1e3, 1e5, 1e7, 1e9, 1e11, 1e13, 1e15, 1e17, 1e19, 1e21, 1e23]

P = np.diag([0] + [1] * 12)

for lambda_ in lambdas:

    X_ridge = np.vstack([
        X,
        np.sqrt(lambda_) * P
    ])

    y_ridge = np.concatenate([
        y,
        np.zeros(13)
    ])

    alpha_ridge, _, _, _ = np.linalg.lstsq(
        X_ridge, y_ridge, rcond=None
    )

    y_pred_ridge = X @ alpha_ridge
    sse_ridge = np.sum((y - y_pred_ridge)**2)
    coefficient_norm = np.linalg.norm(alpha_ridge[1:])

    print(
        f"lambda={lambda_:>8g} | "
        f"SSE={sse_ridge:.3e} | "
        f"||alpha_1:12||={coefficient_norm:.3e}"
    )

lambda_ = 100000

P = np.diag([0] + [1] * 12)

X_ridge = np.vstack([
    X,
    np.sqrt(lambda_) * P
])

y_ridge = np.concatenate([
    y,
    np.zeros(13)
])

alpha_ridge, _, _, _ = np.linalg.lstsq(
    X_ridge, y_ridge, rcond=None
)

y_pred_ridge = X @ alpha_ridge

sse_ridge = np.sum((y - y_pred_ridge)**2)

print("\nRidge coefficients:")
print(alpha_ridge)

print("\nRisdge SSE:")
print(sse_ridge)

penalty_ols = np.sum(alpha_ols[1:]**2)
penalty_ridge = np.sum(alpha_ridge[1:]**2)

print("OLS penalty:", penalty_ols)
print("Ridge penalty:", penalty_ridge)

# ### Plot

x_plot = np.linspace(min(x), max(x), 500)

X_plot = np.column_stack([
    x_plot**i for i in range(13)
])

y_plot_ols = X_plot @ alpha_ols
y_plot_ridge = X_plot @ alpha_ridge

plt.scatter(x, y, label="Data")
plt.plot(x_plot, y_plot_ols, label="OLS")
plt.plot(x_plot, y_plot_ridge, label="Ridge", linestyle='dashed')

plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.savefig("Graphics/plot_task5.pdf", format='pdf', dpi=300, bbox_inches='tight')

plt.show()

# ## Task 6

# ### Parameters

xi17 = 56.0     # ξ17
xi18 = 67.0     # ξ18
xi19 = 82.57    # ξ19

n = 10
alpha = 3

# ### a) Posterior Distribution of $\theta$

# Posterior parameters in shape-rate form

shape = xi17 + 3 * n
rate = n * xi19 + 1 / xi18

# Convert rate to scale

scale = 1 / rate

print(f"shape = {shape}")
print(f"rate = {rate:.8f}")
print(f"scale = {scale:.8f}")

# ### b) Bayes point estimate of $\theta$ associated with the square-error loss function

posterior_mean = shape / rate

print(f"Bayes estimate (squared-error loss): {posterior_mean:.8f}")

# ### c) Bayes point estimate of $\theta$ using the mode of the posterior distribution

posterior_mode = (shape - 1) / rate

print(posterior_mode)

