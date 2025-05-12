
export PoiSpoModel, PoiSpoModelNR

using CSV, DataFrames, LinearAlgebra


"""
    PoiSpoModel(P, T)

Estimate the MLE of the Poisson regression model by a block relaxation algorithm. 

# Positional arguments
- `P :: AbstractMatrix`: `P[i, j]` is the total points team `i` scores againt team `j`.
- `T :: AbstractMatrix`: `T[i, j]` is the total time team `i` plays against team `j`.

# Keyword arguments
- `maxiter   :: Integer`: maximum number of iterations, default is `1000`.
- `tolx      :: Number`: tolerance for iterate convergence, default is `1e-6`.
- `verbose   :: Bool`  : verbose display, default is `false`.
"""
function PoiSpoModel(
        P, T;
        maxiter :: Integer = 1000,
        tolx    :: Number = 1e-6,
        verbose :: Bool = false
    )
    # number of teams
    n = size(P, 1)
    # enforce zero diagonal entries and identifiability constraint d1=0
    for i in 1:n
        P[i, i] = T[i, i] = 0
    end
    # pre-compute row and col sums of P
    prowsum = sum(P, dims = 2) |> vec
    pcolsum = sum(P, dims = 1) |> vec
    # pre-allocate intermediate arrays
    edi = P[:, 1]
    edi = fill!(edi, 1)
    eo  = similar(edi)
    Δx  = similar(eo)
    # main iteration
    iters = 0
    for iter in 1:maxiter
        iters += 1
        # update exp(o_j)
        copy!(Δx, eo)
        mul!(eo, T, edi)
        eo .= prowsum ./ eo
        Δx .-= eo
        nrmchg = norm(Δx)
        # update exp(-d_j)
        copy!(Δx, edi)
        mul!(edi, transpose(T), eo)
        edi .= pcolsum ./ edi
        Δx .-= edi
        nrmchg += norm(Δx)
        # convergence check
        (nrmchg < tolx) && break
    end
    # output
    o = log.(eo)
    d = -log.(edi)
    # fix identifiability
    # o .-= d[1]
    # d .-= d[1]
    o, d, iters
end


function log_likelihood(params, goal_matrix, match_duration)
    num_teams, _ = size(goal_matrix)
    α = params[1:num_teams]  # offensive strength
    β = params[num_teams+1:end]  # defensive strength
    logL = 0.0
    for i in 1:num_teams
        for j in 1:num_teams
            if i != j  
                λ_ij = exp(α[i] - β[j]) * match_duration[i, j]
                logL += goal_matrix[i, j] * (α[i] - β[j]) - λ_ij - loggamma(goal_matrix[i, j] + 1)
            end
        end
    end
    return logL
end

function compute_gradient_hessian(params, goal_matrix, match_duration)
    num_teams, _ = size(goal_matrix)
    α = params[1:num_teams]
    β = params[num_teams+1:end]
    grad = zeros(2 * num_teams)
    H = zeros(2 * num_teams, 2 * num_teams)
    for i in 1:num_teams
        for j in 1:num_teams
            if i != j
                λ_ij = exp(α[i] - β[j]) * match_duration[i, j]
                grad[i] += goal_matrix[i, j] - λ_ij
                grad[j + num_teams] += -(goal_matrix[i, j] - λ_ij)
                H[i, i] -= λ_ij
                H[j + num_teams, j + num_teams] -= λ_ij
                H[i, j + num_teams] += λ_ij
                H[j + num_teams, i] += λ_ij
            end
        end
    end
    return grad, H
end

# Backtracking Line Search
function line_search(params, direction, goal_matrix, match_duration; α=1.0, c=1e-4, ρ=0.5, max_iter=20)
    num_teams, _ = size(goal_matrix)
    old_likelihood = log_likelihood(params, goal_matrix, match_duration)
    for _ in 1:max_iter
        new_params = params + α * direction
        new_likelihood = log_likelihood(new_params, goal_matrix, match_duration)
        if new_likelihood > old_likelihood + c * α * dot(direction, direction)
            return α 
        end
        α *= ρ  
    end
    return α  
end

# Newton-Raphson 
function PoiSpoModelNR(goal_matrix, match_duration; tol=1e-6, max_iter=10000)
    num_teams, _ = size(goal_matrix)
    params = zeros(2 * num_teams)  
    for iter in 1:max_iter
        grad, H = compute_gradient_hessian(params, goal_matrix, match_duration)
        if norm(grad) < tol
#             println("Converged after $iter iterations.")
            return params
        end
        direction = -H \ grad  
        step_size = line_search(params, direction, goal_matrix, match_duration)
        params += step_size * direction
    end
    return params
end