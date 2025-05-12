export DirichletMinka, DirichletNR, DirichletMM

using LinearAlgebra


# MM algorithm
function _invdigamma(y::Float64, x, digamma_, trigamma_)
    x_old = x
    x_new = x_old
    delta = Inf
    iteration = 0
    while delta > 1e-5 && iteration < 3
        iteration += 1
#         trigamma_ = trigamma(x_old)
        x_new = max(x_old - (digamma_ - y) / trigamma_, 1e-12)
        delta = abs(x_new - x_old)
        x_old = x_new
        digamma_  = digamma(x_new)
        trigamma_ = trigamma(x_new)
    end
    return x_new, digamma_, trigamma_
end

function DirichletMM(x::Matrix)
    T = eltype(x)
    (m, p) = size(x)
    avglog = mean(log, x, dims=2)
    lambda = ones(T, m)
    dg = Vector{Float64}(undef, m)
    eps0 = 1e-6 / p
    iters = 0
    digammas = digamma.(lambda)
    trigammas = trigamma.(lambda)
    
    for iteration = 1:1000
        lambda_sum = sum(lambda)  # 并行求和
        c = digamma(lambda_sum)
        iters += 1 
        for i = 1:m
            @inbounds avglogi = c + avglog[i]
            @inbounds dg[i] = avglogi - digammas[i]
            @inbounds lambda[i], digammas[i], trigammas[i] = _invdigamma(avglogi, lambda[i], digammas[i], trigammas[i])
        end
        if norm(dg) < eps0
            break
        end
    end
    return (lambda, iters)
end

# minka's method
function DirichletMinka(x::Matrix)    
    # This function directly overrides α
    (elogp, iters) = (mean(log, x, dims=2), 0)
    (K, p) = size(x)
    g = Vector{Float64}(undef, K)
    iq = Vector{Float64}(undef, K)
    α = ones(K)  
    α0 = sum(α)
    dg = zeros(Float64, K)
    t = 0
    eps0 = 1e-6/p
    converged = false
    while !converged && t < 1000
        t += 1
        # compute gradient & Hessian
        # (b is computed as well)
        digam_α0 = digamma(α0)
        iz = 1.0 / trigamma(α0)
        gnorm = 0.
        b = 0.
        iqs = 0.
        for k = 1:K
            @inbounds ak = α[k]
            @inbounds g[k] = gk = digam_α0 - digamma(ak) + elogp[k]
            @inbounds iq[k] = - 1.0 / trigamma(ak)
            @inbounds b += gk * iq[k]
            @inbounds iqs += iq[k]
#             dg[k] = gk
        end
        b /= (iz + iqs)
        # update α
        for k = 1:K
            @inbounds α[k] -= (g[k] - b) * iq[k]
            @inbounds if α[k] < 1.0e-12
                α[k] = 1.0e-12
            end
        end
        α0 = sum(α)
        # determine convergence
        converged = norm(g) < eps0
    end
    return (α, t)
end

using SpecialFunctions, LinearAlgebra

# Compute the gradient of the Dirichlet log-likelihood
function dirichlet_gradient(α, elogp)
    sum_alpha = sum(α)
    ψ_sum_alpha = digamma(sum_alpha)  # Digamma function at sum(α)
    return elogp .+ ψ_sum_alpha .- digamma.(α)
end

# Compute the Hessian matrix explicitly
function dirichlet_hessian(α)
    K = length(α)
    sum_alpha = sum(α)
    H = zeros(K, K)

    ψ_prime_sum_alpha = trigamma(sum_alpha)  # Trigamma function ψ'(∑α_j)

    for i in 1:K
        H[i, i] = trigamma(α[i]) - ψ_prime_sum_alpha  # Diagonal elements
        for j in 1:K
            if i != j
                H[i, j] = -ψ_prime_sum_alpha  # Off-diagonal elements
            end
        end
    end
    return H
end

# Newton Raphson Algorithm (standard)
function DirichletNR(x::Matrix)    
    # This function directly overrides α
    (elogp, iters) = (mean(log, x, dims=2), 0)
    (K, p) = size(x)
    g = Vector{Float64}(undef, K)
    α = ones(K)  
    dg = zeros(Float64, K)
    t = 0
    eps0 = 1e-6/p
    converged = false
    while !converged && t < 1000
        t += 1
        # compute gradient & Hessian
        hess = dirichlet_hessian(α)
        grad = dirichlet_gradient(α, elogp)
        
        Δα = hess \ reshape(grad, :, 1)  # Convert grad to (16,1)
        α_new = α + Δα
        α_new = max.(α_new, 1e-12)
        
        g = digamma(sum(α_new)) .- digamma.(α_new) .+ elogp
        
        α = α_new
        # determine convergence
        converged = norm(g) < eps0
    end
    return (vec(α), t)
end