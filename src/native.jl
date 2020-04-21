using Statistics, LinearAlgebra

# bonds of a petersen graph
const L1 = [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10),
    (1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (6, 8),
    (8, 10), (7, 10), (7, 9), (6, 9)]

# disconnected bonds of a petersen graph
const L2 = [(1, 3), (1, 4), (1, 7), (1, 8), (1, 9),
    (1, 10), (2, 4), (2, 5), (2, 6), (2, 8), (2, 9),
    (2, 10), (3, 5), (3, 6), (3, 7), (3, 9), (3, 10),
    (4, 6), (4, 7), (4, 8), (4, 10), (5, 6), (5, 7),
    (5, 8), (5, 9), (6, 7), (6, 10), (7, 8), (8, 9),
    (9, 10)]

function myvar_and_mean(v)
    mv  = Statistics.mean(v)
    sum((v .- mv).^2)./(length(v)-1), mv
end

function relu(x::T) where T
    x > 0 ? x : zero(T)
end

function embedding_loss(x)
    a = [LinearAlgebra.norm(x[:,i]-x[:,j]) for (i, j) in L1]
    b = [LinearAlgebra.norm(x[:,i]-x[:,j]) for (i, j) in L2]
    va, ma = myvar_and_mean(a)
    vb, mb = myvar_and_mean(b)
    va + vb + exp(relu(-mb + ma + 0.1)) - 1
end
