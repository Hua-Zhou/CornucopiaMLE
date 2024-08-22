using Pkg
Pkg.activate(pwd())

using LinearAlgebra, Random, Distributions, StatsBase, SpecialFunctions

function mle_Cauchy(x::Vector)
#
  (m, iters) = (length(x), 0)
  w = similar(x) # weight vector
  (mu, sigma) = (median(x), iqr(x) / 2) # Wikipedia initial values
  (old_mu, old_sigma) = (mu, sigma)
  for iteration = 1:100 # MM updates
    iters = iters + 1
    @. w = 1 / (1 + ((x - mu) / sigma)^2)
    mu = dot(w, x) / sum(w) # MM update of mu
    s = sum(w .* (x .- mu).^2)
    sigma = sqrt(2s / m) # MM update of sigma
#     f = loglikelihood(Cauchy(mu, sigma), x)
#     println(iteration," ",f," ",mu," ",sigma)
    if abs(old_mu - mu) + abs(old_sigma -  sigma) < 1.0e-6 # convergence
      break
    end
    (old_mu, old_sigma) = (mu, sigma)
  end
  return (mu, sigma, iters)
end

function mle_gumbel(x::Vector)
  (m, avg, iters) = (length(x), mean(x), 0)
  alpha = 1 / sqrt(6 * var(x) / pi^2)
  for iteration = 1:500
    iters = iters + 1
    df = m / alpha - m * avg
    (s1, s2, s3) = (0.0, 0.0, 0.0)
    f = m * log(alpha) - m * alpha * avg
    for i = 1:m
      c = exp(-alpha * x[i])
      s1 = s1 + c
      s2 = s2 + c * x[i]
      s3 = s3 + c * x[i]^2
    end
    f = f - m * log(s1)
    df = df + m * s2 / s1
    d2f = -m / alpha^2 - m * s3 / s1 + df^2
    alpha = alpha - df / d2f # Newton update
#     println(iteration," ",f," ",alpha," ",df)
    if abs(df) < 1e-6
      break
    end
  end
  beta = 1 / alpha
  g(y) = exp(-y / beta)
  mu = - beta * log(mean(g, x))
  return (beta, mu, iters)
end

function mle_gumbel2(x::Vector)
  (m, avg, iters) = (length(x), mean(x), 0)
  v = (maximum(x) - minimum(x))^2 / 4
  alpha = 1 / (sqrt(6 * var(x) / pi^2))
  for iteration = 1:500
    iters = iters + 1
    (s1, s2) = (0.0, 0.0)
    for i = 1:m
      c = exp(-alpha * x[i])
      s1 = s1 + c
      s2 = s2 + c * x[i]
    end
    b = s2 / s1 - avg + alpha * v
    df = m / alpha - m * avg + m * s2 / s1
    alpha = (b + sqrt(b^2 + 4v)) / (2v)
#     f = loglikelihood(Gumbel(1 / alpha), x)
#     println(iteration," ",f," ",1 / alpha," ",df)
    if abs(df) < 1e-6
      break
    end
  end
  beta = 1 / alpha
  g(y) = exp(-y / beta)
  mu = -beta * log(mean(g, x))
  return (beta, mu, iters)
end

function mle_yule_simon(x::Vector)
  (m, avg, iters) = (length(x), mean(x), 0)
  rho = max(avg / (avg - 1), 1.0)
  (f, df) = (0.0, 0.0)
  for iteration = 1:100
    iters = iters + 1
    f = m * log(rho)
    s = 0.0
    for i = 1:m
      s = s - digamma(x[i] + rho  + 1) + digamma(rho + 1)
      f = f + loggamma(x[i]) + loggamma(rho + 1) - loggamma(x[i] + rho + 1)
    end
    df = m / rho + s  
    rho = - m / s
#     println(iteration," ",f," ",rho," ",df)
    if abs(df) < 1e-6
      break
    end
  end
  return (rho, iters)
end

function mle_logarithmic(x::Vector)
  (m, avg, iters) = (length(x), mean(x), 0)
  q = 1.0
  for iteration = 1:100
    iters = iters + 1
    eq = exp(q)
    b = eq / ((1 + eq) * log(1 + eq))
    f = -log(log(1 + eq)) + avg * (q - log(1 + eq))
    df = -b + avg - avg * eq / (1 + eq)
    q = q + 4 * df / avg 
#     println(iteration," ",f," ",q," ",df)
    if abs(df) < 1e-6
      break
    end
  end
  return (exp(q) / (1 + exp(q)), iters)
end

# function mle_logarithmic2(x::Vector)
#   (m, avg, iters) = (length(x), mean(x), 0)
#   q = 1.0
#   for iteration = 1:100
#     iters = iters + 1
#     eq = exp(q)
#     b = eq / ((1 + eq) * log(1 + eq))
#     f = -log(log(1 + eq)) + avg * (q - log(1 + eq))
#     df = -b + avg - avg * eq / (1 + eq)
#     q = -(b - avg / 2) / ((avg * (eq - 1) / (eq + 1) / (2q)))
#     println(iteration," ",f," ",q," ",df)
#     if abs(df) < 1e-6
#       break
#     end
#   end
#   return (exp(q) / (1 + exp(q)), iters)
# end

function mle_weibull(x::Vector)
  (kappa, old_kappa, lambda) = (1.0, 1.0, 1.0)
  (avg, avglog, iters) = (mean(x), mean(log.(x)), 0)
  for iteration = 1:100
    iters = iters + 1
    a = sum(x.^kappa .* log.(x))
    b = sum(x.^kappa)
    kappa = 1 /(a / b - avglog)
    lambda = mean(x.^kappa)^(1 / kappa)
#     f = loglikelihood(Weibull(kappa, lambda), x)
#     println(iteration," ",f)
    if abs(kappa - old_kappa) < 1e-6
      break
    else
      old_kappa = kappa
    end
  end
  return (kappa, lambda, iters)   
end

function mle_rice(x::Vector{Float64})
  (m, iters) = (length(x), 0)
  sumsq = mean(x.^2)
  w = similar(x)
  (nu, old_nu) = (1.0, 1.0)
  (sigmasq, old_sigmasq) = (1.0, 1.0)
  for iteration = 1:500
    iters = iters + 1
    c = nu / sigmasq
    @. w = besseli(1.0, c * x) / besseli(0.0, c * x)
    c = dot(w, x)
    nu = c / m
    sigmasq = (sumsq + nu^2) / 2 - nu * c / m
#     f = loglikelihood(Rician(nu, sqrt(sigmasq)), x)
#     println(iteration," ",f)
    if abs(nu - old_nu) + abs(sigmasq - old_sigmasq) < 1e-6
      break
    else
      (old_nu, old_sigmasq) = (nu, sigmasq)
    end
  end 
  return (nu, sigmasq, iters)  
end

function mle_dirichlet(x::Matrix)
  (m, p) = size(x)
  (avglog, iters) = (mean(log.(x), dims = 2), 0)
  (lambda, df) = (ones(m), zeros(m))
  for iteration = 1:100
    iters = iters + 1
    c = digamma(sum(lambda))
    for i = 1:m
      df[i] = p * (c - digamma(lambda[i]) + avglog[i])
      lambda[i] = invdigamma(c + avglog[i])
    end
#     f = loglikelihood(Dirichlet(lambda), x)
#     println(iteration," ",f," ",norm(df))
    if norm(df) < 1e-6
      break
    end  
  end
  return (lambda, iters)
end

function mle_negative_binomial(x::Vector)
  (m, iters) = (length(x), 0)
  (p, r) = (0.5, 1.0)
  (old_p, old_r) = (0.5, 1.)
  avg = mean(x)
  for iteration = 1:500
    iters = iters + 1
    s = 0.0
    for i = 1:m
      for j = 0:(x[i] - 1)
        s = s + r / (r + j)
      end
    end
    r = -s / (m * log(p)) # MM update
    p = r / (r + avg) # MLE update
#     f = loglikelihood(NegativeBinomial(r,p), x)
#     println(iteration," ",f)
    if abs(p - old_p) + abs(r - old_r) < 1e-6
      break
    else
      (old_p, old_r) = (p, r)
    end
  end
  return (p, r, iters)   
end

function mle_negative_binomial2(x::Vector)
  (m, iters) = (length(x), 0)
  (avg, ssq) = (mean(x), var(x))
  (p, r) = (avg / ssq, avg^2 / (ssq - avg)) # MOM estimates
  if r <= 0.0
    r =  1.0
  end
  (old_p, old_r) = (p, r)
  avg = mean(x)
  for iteration = 1:500
    iters = iters + 1
    df = m * log(p)
    d2f = 0.0
    for i = 1:m
      for j = 0:(x[i] - 1)
        d = 1 / (r + j)
        df = df + d
        d2f = d2f - d^2 
      end
    end
    r = r - df / d2f # Newton update 
    p = r / (r + avg) # MLE update
#     f = loglikelihood(NegativeBinomial(r, p), x)
#     println(iteration,"  ",f)
    if abs(p - old_p) + abs(r - old_r) < 1.0e-6
      break
    end
    (old_p, old_r) = (p, r)
  end
  return (p, r, iters)   
end

function mle_inverse_gamma(x::Vector)
  (m, iters) = (length(x), 0)
  (avglog, avginverse) = (mean(log.(x)), mean(1 ./ x))
  (alpha, old_alpha) = (mean(x)^2 / var(x) + 2, 1.0)
  (beta, old_beta) = (1.0, 1.0)
  for iteration = 1:100
    iters = iters + 1
    beta = alpha / avginverse
    alpha = invdigamma(log(beta) - avglog)
#     f = loglikelihood(InverseGamma(alpha, beta), x)
#     println(iteration," ",f," ",alpha)
    if abs(alpha - old_alpha) + abs(beta - old_beta) < 1e-6
      break
    else
      (old_alpha, old_beta) = (alpha, beta)
    end  
  end
  return (alpha, beta, iters)
end

function mle_gamma(x::Vector)
  (avg, avglog, iters) = (mean(x), mean(log.(x)), 0)
  d = log(avg) - avglog
  alpha = (3 - d + sqrt((3 - d)^2 + 24d)) / (12d)
  (old_alpha, beta, old_beta) = (0.0, 0.0, 0.0)
  for iteration = 1:100
    iters = iters + 1
    beta = alpha / avg
    alpha = invdigamma(log(beta) + avglog)
#     f = loglikelihood(Gamma(alpha, 1 / beta), x)
#     println(iteration," ",f," ",alpha)
    if abs(alpha - old_alpha) + abs(beta - old_beta) < 1e-6
      break
    else
      (old_alpha, old_beta) = (alpha, beta)
    end  
  end
  return (alpha, beta, iters)
end

function logarithmic_deviate(p, n)
  x = zeros(Int, n)
  mu = - p / (log(1 - p) * (1 - p))
  v = -(p^2 + p * log(1  - p)) / ((1 - p) * log( 1 - p))^2
  x[1] = round(Int, mu)
  for i = 1:(n - 1)
    u = rand(2)
    if u[1] < 1 / 2
      if u[2] < (x[i] - 1) / (p * x[i])
        x[i + 1] = x[i] - 1
      else
        x[i + 1] = x[i]
      end 
    else
      if u[2] <  (p / (x[i] + 1)) * x[i] 
        x[i + 1] = x[i] + 1
      else
        x[i + 1] = x[i]
      end
    end
  end
  return x
end

function yule_simon_deviate(rho, n)
  x = zeros(Int, n)
  mu = rho / (rho - 1)
  x[1] = max(round(Int, mu), 1)
  for i = 1:(n - 1)
    u = rand(2)
    if x[i] == 1
      if u[1] < 1 / (2 * (rho + 2))
        x[i + 1] = 2
      else
        x[i + 1] = 1
      end   
    elseif u[1] < 1 / 2
      if u[2] < min((x[i] + rho) / (x[i] - 1), 1.0)
        x[i + 1] = x[i] - 1
      else
        x[i + 1] = x[i]
      end
    else
      if u[2] <  x[i] / (x[i] + rho + 1) 
        x[i + 1] = x[i] + 1
      else
        x[i + 1] = x[i]
      end
    end
  end
  return x
end

Random.seed!(12345);
for trial = 1:2
println("trial = ",trial)
#
# Yule-Simon distribution
#
(m, rho) = (1000, 3.0);
x = yule_simon_deviate(rho, m);
@time (rho, iters) = mle_yule_simon(x)
println("Yule-Simon & ",rho," & ",iters)
#
# Negative binomial distribution
#
(m, p, r) = (1000, 0.25, 5.0);
x = rand(NegativeBinomial(r, p), m);
avg = mean(x);
ssq = var(x);
@time (p, r, iters) = mle_negative_binomial(x)
println("Negative binomial & ",p," ",r," & ",iters)
@time (p, r, iters) = mle_negative_binomial2(x)
println("Negative binomial & ",p," ",r," & ",iters)
(p, r) = (avg/ssq, avg^2/(ssq-avg))
#
# Logarithmic distribution
#
(m, p, q) = (1000, 1, 0.4);
x = logarithmic_deviate(q, m);
@time (q, iters) = mle_logarithmic(x)
println("logarithmic & ",q," & ",iters)
#
# Cauchy distribution
#
(m, p) = (1000, 2);
(mu, sigma) = (1.0, 1.0)
x = rand(Cauchy(mu, sigma), m);
@time (mu, sigma, iters) = mle_Cauchy(x)
println("Cauchy & ",mu,"  ",sigma," & ",iters)
#
# Gumbel distribution
#
(m, p) = (1000, 2);
(beta, mu) = (2.0, 0.5);
x = rand(Gumbel(mu, beta), m);
@time (beta, mu, iters) = mle_gumbel(x)
println("Gumbel & ",beta," ",mu," & ",iters)
@time (beta, mu, iters) = mle_gumbel2(x)
println("Gumbel & ",beta," ",mu," &  ",iters)
sqrt(6 * var(x) / pi^2)
#
# Weibull distribution
#
(m, p) = (1000, 2);
(kappa, lambda) = (2.0, 3.0);
x = rand(Weibull(kappa, lambda), m);
@time (kappa, lambda, iters) = mle_weibull(x)
println("Weibel & ",kappa," ",lambda," & ",iters)
#
# Rice distribution
#
(m, p) = (1000, 2);
(nu, sigmasq) = (2.0, 3.0);
x = rand(Rician(nu, sqrt(sigmasq)), m);
@time (nu, sigmasq, iters) = mle_rice(x)
println("Rice & ",nu," ",sigmasq," & ",iters)
#
# Dirichlet distribution
#
(m, p) = (1000, 3);
lambda = [1/3, 1/3, 1/3];
x = rand(Dirichlet(lambda), m);
@time (lambda, iters) = mle_dirichlet(x)
println("Dirichlet & ",lambda," & ",iters)
#
# Inverse gamma distribution
#
(m, p) = (1000, 2);
(alpha, beta) = (2.0, 3.0);
x = rand(InverseGamma(alpha, beta), m);
@time (alpha, beta, iters) = mle_inverse_gamma(x)
println("Inverse gamma & ",alpha," ",beta," & ",iters)
#
# Gamma distribution
#
(m, p) = (1000, 2);
(alpha, beta) = (2.0, 3.0);
x = rand(Gamma(alpha, 1 / beta), m);
@time (alpha, beta, iters) = mle_gamma(x)
println("Gamma & ",alpha," ",beta," & ",iters)
end

