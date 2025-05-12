export Ken, OptTransport, Sinkhorn

using LinearAlgebra
using JuMP, HiGHS

function Sinkhorn(a::Vector, b::Vector, C::AbstractMatrix, ε; max_iter::Int=10000, tol=1e-6)
    n, m = size(C)
    
    K = exp.(-C / ε)
    u = ones(n)
    v = ones(m)
    
    iters = 0
    for _ in 1:max_iter
        iters += 1
        u_prev = copy(u)
        u = a ./ (K * v)  
        v = b ./ (K' * u)  
        if norm(u - u_prev, Inf) < tol
            break
        end
    end
    P = Diagonal(u) * K * Diagonal(v)
#     println("Converged after $iters iterations.")
    return P, iters
end

function Ken(r::Vector, c::Vector, M::AbstractMatrix; max_iter::Int=10000, tol::Float64=1e-10)
    n, m = size(M)
    a = ones(n)
    b = ones(m)
    a_new = similar(a)
    b_new = similar(b)
    iters = 0
    for iter in 1:max_iter
        iters += 1
        mul!(a_new, M, b)
        a_new .= r ./ a_new
        mul!(b_new, transpose(M), a_new)
        b_new .= c ./ b_new
        if maximum(abs.(a_new .- a)) < tol && maximum(abs.(b_new .- b)) < tol
            println("Converged after $iter iterations.")
            a = a_new
            b = b_new
            return Diagonal(a) * M * Diagonal(b), iters
        end
        copy!(a, a_new)
        copy!(b, b_new)
    end
    println("Reached maximum iterations.")
    return Diagonal(a) * M * Diagonal(b), iters
end

function OptTransport(a, b, C)
    m, n = length(a), length(b)
    model = Model(HiGHS.Optimizer)
    set_silent(model) 
    @variable(model, X[1:m, 1:n] ≥ 0)
    @objective(model, Min, sum(C[i, j] * X[i, j] for i in 1:m, j in 1:n))
    @constraint(model, [i in 1:m], sum(X[i, j] for j in 1:n) == a[i])
    @constraint(model, [j in 1:n], sum(X[i, j] for i in 1:m) == b[j])
    optimize!(model)
    return value.(X), 1
end