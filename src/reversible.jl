"""the mean value"""
@i function mean(out!, x)
    anc ← zero(out!)
    for i=1:length(x)
        anc += identity(x[i])
    end
    mulint(out!, length(x))
end

"""
    var_and_mean_sq(var!, mean!, sqv)

the variance and mean value from squared values.
"""
@i function var_and_mean_sq(var!, mean!, sqv::AbstractVector{T}) where T
    sqmean ← zero(mean!)
    @inbounds for i=1:length(sqv)
        mean! += sqv[i] ^ 0.5
        var! += identity(sqv[i])
    end
    divint(mean!, length(sqv))
    divint(var!, length(sqv))
    sqmean += mean! ^ 2
    var! -= identity(sqmean)
    sqmean -= mean! ^ 2
    mulint(var!, length(sqv))
    divint(var!, length(sqv)-1)
end

"""
Squared distance of two vertices.
"""
@i @inline function sqdistance(dist!, x1::AbstractVector{T}, x2::AbstractVector) where T
    @invcheckoff @inbounds for i=1:length(x1)
        x1[i] -= identity(x2[i])
        dist! += x1[i] ^ 2
        x1[i] += identity(x2[i])
    end
end

"""The loss of graph embedding problem."""
@i function embedding_loss(out!::T, x) where T
    v1 ← zero(T)
    m1 ← zero(T)
    v2 ← zero(T)
    m2 ← zero(T)
    diff ← zero(T)
    d1 ← zeros(T, length(L1))
    d2 ← zeros(T, length(L2))
    @routine @invcheckoff begin
        for i=1:length(L1)
            @inbounds sqdistance(d1[i], x[:,L1[i][1]],x[:,L1[i][2]])
        end
        for i=1:length(L2)
            @inbounds sqdistance(d2[i], x[:,L2[i][1]],x[:,L2[i][2]])
        end
        var_and_mean_sq(v1, m1, d1)
        var_and_mean_sq(v2, m2, d2)
        m1 -= identity(m2)
        m1 += identity(0.1)
    end
    out! += identity(v1)
    out! += identity(v2)
    if (m1 > 0, ~)
        # to ensure mean(v2) > mean(v1)
        # if mean(v1)+0.1 - mean(v2) > 0, punish it.
        out! += exp(m1)
        out! -= identity(1)
    end
    ~@routine
end
