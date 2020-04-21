export get_hessian_nilang, get_hessian_reversediff

function get_hessian_nilang(params0::AbstractArray{T}) where T
    N = length(params0)
    params = Dual.(params0, zero(T))
    hes = zeros(T, N, N)
    for i=1:N
        @inbounds i !== 1 && (params[i-1] = Dual(params0[i-1], zero(T)))
        @inbounds params[i] = Dual(params0[i], one(T))
        res = NiLang.AD.gradient(Val(1), embedding_loss, (Dual(0.0, 0.0), params))[2]
        hes[:,i] .= vec(ForwardDiff.partials.(res, 1))
    end
    hes
end
