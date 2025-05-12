export MMLS, Marquardt

using LinearAlgebra

# function MMLS_KL(y::Vector{T}, X::Matrix{T}, d) where T <: Real
#   ((n, p), iters) = (size(X), 0)
#   XTX = transpose(X) * X
#   XTy = transpose(X) * y
#   mu = d * norm(XTX)
#   C = cholesky(XTX + mu * I)
# #    C = qr(XTX + mu * I)
#   beta = zeros(T, p)
#   for iter = 1:1000
#     iters = iters + 1   
#     beta = C \ (XTy + mu * beta)
# #     println(iter," ",norm(y - X * beta))
#     if norm(XTX * beta - XTy) < 1.0e-4 # gradient test
#       break
#     end
#   end
#   return (iters, beta) 
# end


"""
    MMLS(y, X, d)

Solve least squares by an MM algorithm.

# Positional arguments
- `y :: AbstractVector`: response vector.
- `X :: AbstractMatrix`: feature matrix.
- `d :: Number`: condition parameter `μ = d * norm(X'X)`.

# Keyword arguments
- `maxiter :: Integer`: maximum number of iterations, default is `1000`.
- `tol∇    :: Number`: convergence tolerance in gradient, default is `1e-4`.
"""
function MMLS(
        y :: Vector{T}, 
        X :: Matrix{T}, 
        d :: T;
        maxiter :: Integer = 1000,
        tol∇    :: Number = 1e-4
        ) where T <: Real
    ((n, p), iters) = (size(X), 0)
    # pre-compute
    XTX = transpose(X) * X
    XTy = transpose(X) * y
    μ   = d * norm(XTX)
    C   = cholesky(XTX + μ * I)
    # allocate intermediate arrays
    β = zeros(T, p)
    storage_p = similar(β)
    # MM iteration
    for iter in 1:maxiter
        iters = iters + 1   
        # beta = C \ (XTy + mu * beta)
        storage_p .= XTy .+ μ .* β
        ldiv!(β, C, storage_p)
        # println(iter," ",norm(y - X * beta))
        # gradient test
        mul!(copyto!(storage_p, XTy), XTX, β, one(T), -one(T))
        (norm(storage_p) < tol∇) && break
    end
    # output
    return (iters, β)
end

function Marquardt(y::Vector{T}, X::Matrix{T}, d) where T <: Real
  ((n, p), iters) = (size(X), 1)
  XTX = transpose(X) * X
  XTy = transpose(X) * y
  mu = d * norm(XTX) 
  C = cholesky(XTX + mu * I)
  beta = C \ XTy
  return (iters, beta)
end