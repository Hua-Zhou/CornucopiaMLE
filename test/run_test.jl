using BenchmarkTools, Bessels, DataFrames, Distributions, PrettyTables, StatsBase, SpecialFunctions
using LinearAlgebra, Random

Random.seed!(12345)

# result containers
den = String[]
par = Vector{Float64}[]
est = Vector{Float64}[]
its = Int[]
sec = Float64[]

#
# Yule-Simon distribution
#
push!(den, "Yule-Simon")
(m, rho) = (1000, 3.0)
push!(par, [rho])
x = yule_simon_deviate(rho, m)
(rho, iters) = mle_yule_simon(x)
println("Yule-Simon & ", rho, " & ", iters)
push!(est, [rho])
bm = @benchmark mle_yule_simon($x)
display(bm)
push!(its, iters)
push!(sec, median(bm.times) / 1e6)

#
# Negative binomial distribution
#
push!(den, "Negative binomial")
push!(den, "Negative binomial")
(m, p, r) = (1000, 0.25, 5.0);
push!(par, [p, r])
push!(par, [p, r])
x = rand(NegativeBinomial(r, p), m);
avg = mean(x);
ssq = var(x);
@time (p, r, iters) = mle_negative_binomial(x)
println("Negative binomial & ", p, " ", r, " & ", iters)
push!(est, [p, r])
push!(its, iters)
bm = @benchmark mle_negative_binomial($x)
display(bm)
push!(sec, median(bm.times) / 1e6)
@time (p, r, iters) = mle_negative_binomial2(x)
push!(est, [p, r])
push!(its, iters)
println("Negative binomial & ", p, " ", r, " & ", iters)
bm = @benchmark mle_negative_binomial2($x)
display(bm)
push!(sec, median(bm.times) / 1e6)
(p, r) = (avg/ssq, avg^2/(ssq-avg))

#
# Logarithmic distribution
#
push!(den, "Logarithmic")
(m, p, q) = (1000, 1, 0.4);
push!(par, [q])
x = logarithmic_deviate(q, m);
@time (q, iters) = mle_logarithmic(x)
push!(est, [q])
push!(its, iters)
println("logarithmic & ", q, " & ", iters)
bm = @benchmark mle_logarithmic($x)
display(bm)
push!(sec, median(bm.times) / 1e6)

#
# Cauchy distribution
#
push!(den, "Cauchy")
(m, p) = (1000, 2);
(mu, sigma) = (1.0, 1.0)
push!(par, [mu, sigma])
x = rand(Cauchy(mu, sigma), m);
@time (mu, sigma, iters) = mle_Cauchy(x)
push!(est, [mu, sigma])
push!(its, iters)
println("Cauchy & ", mu, "  ", sigma, " & ", iters)
bm = @benchmark mle_Cauchy($x)
display(bm)
push!(sec, median(bm.times) / 1e6)

#
# Gumbel distribution
#
push!(den, "Gumbel")
push!(den, "Gumbel")
(m, p) = (1000, 2)
(beta, mu) = (2.0, 0.5)
push!(par, [beta, mu])
push!(par, [beta, mu])
x = rand(Gumbel(mu, beta), m);
@time (beta, mu, iters) = mle_gumbel(x)
push!(est, [beta, mu])
push!(its, iters)
println("Gumbel & ", beta, " ", mu, " & ", iters)
bm = @benchmark mle_gumbel($x)
display(bm)
push!(sec, median(bm.times) / 1e6)
@time (beta, mu, iters) = mle_gumbel2(x)
push!(est, [beta, mu])
push!(its, iters)
println("Gumbel & ", beta, " ", mu, " &  ", iters)
bm = @benchmark mle_gumbel2($x)
display(bm)
push!(sec, median(bm.times) / 1e6)
sqrt(6 * var(x) / pi^2)

#
# Weibull distribution
#
push!(den, "Weibull")
(m, p) = (1000, 2);
(kappa, lambda) = (2.0, 3.0);
push!(par, [kappa, lambda])
x = rand(Weibull(kappa, lambda), m);
@time (kappa, lambda, iters) = mle_weibull(x)
push!(est, [kappa, lambda])
push!(its, iters)
println("Weibull & ", kappa, " ", lambda, " & ", iters)
bm = @benchmark mle_weibull($x)
display(bm)
push!(sec, median(bm.times) / 1e6)

#
# Rice distribution
#
push!(den, "Rice")
(m, p) = (1000, 2)
(nu, sigmasq) = (2.0, 3.0)
push!(par, [nu, sigmasq])
x = rand(Rician(nu, sqrt(sigmasq)), m)
@time (nu, sigmasq, iters) = mle_rice(x)
push!(est, [nu, sigmasq])
push!(its, iters)
println("Rice & ", nu, " ", sigmasq, " & ", iters)
bm = @benchmark mle_rice($x)
display(bm)
push!(sec, median(bm.times) / 1e6)

#
# Dirichlet distribution
#
push!(den, "Dirichlet")
(m, p) = (1000, 3);
lambda = [1 / 3, 1 / 3, 1 / 3];
push!(par, lambda)
x = rand(Dirichlet(lambda), m);
@time (lambda, iters) = mle_dirichlet(x)
push!(est, lambda)
push!(its, iters)
println("Dirichlet & ", lambda, " & ", iters)
bm = @benchmark mle_dirichlet($x)
display(bm)
push!(sec, median(bm.times) / 1e6)

#
# Inverse gamma distribution
#
push!(den, "Inverse gamma")
(m, p) = (1000, 2)
(alpha, beta) = (2.0, 3.0)
push!(par, [alpha, beta])
x = rand(InverseGamma(alpha, beta), m);
@time (alpha, beta, iters) = mle_inverse_gamma(x)
push!(est, [alpha, beta])
push!(its, iters)
println("Inverse gamma & ", alpha, " ", beta, " & ", iters)
bm = @benchmark mle_inverse_gamma($x)
display(bm)
push!(sec, median(bm.times) / 1e6)

#
# Gamma distribution
#
push!(den, "Gamma")
push!(den, "Gamma")
(m, p) = (1000, 2)
(alpha, beta) = (2.0, 3.0)
push!(par, [alpha, beta])
push!(par, [alpha, beta])
x = rand(Gamma(alpha, 1 / beta), m);
# method 1 based on Stirling's approximation
@time (alpha, beta, iters) = mle_gamma(x)
push!(est, [alpha, beta])
push!(its, iters)
println("Gamma & ", alpha, " ", beta, " & ", iters)
bm = @benchmark mle_gamma($x)
display(bm)
push!(sec, median(bm.times) / 1e6)
# method 1 based on QLB MM algorithm
@time (alpha, beta, iters) = mle_gamma1(x)
push!(est, [alpha, beta])
push!(its, iters)
println("Gamma & ", alpha, " ", beta, " & ", iters)
bm = @benchmark mle_gamma1($x)
display(bm)
push!(sec, median(bm.times) / 1e6)

results = DataFrame(
    Density=den,
    Parameters=par,
    Estimate=est,
    Iterations=its,
    Time=sec
)
# display(results)
pretty_table(
    results,
    header=["Density", "Parameters", "Estimate", "Iterations", "Time (ms)"],
    formatters=(v, i, j) -> (j == 2 || j == 3) ? round.(v, digits=3) : (j == 5 ? round(v, digits=3) : v)
)

# LaTeX table for the paper
pretty_table(
    results,
    header=["Density", "Parameters", "Estimate", "Iterations", "Time (ms)"],
    formatters=(v, i, j) -> (j == 2 || j == 3) ? round.(v, digits=3) : (j == 5 ? round(v, digits=3) : v),
    backend = Val(:latex),
    vlines = :all,
    alignment = :c, 
    compact_printing = false
)