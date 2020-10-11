"""the mean value"""
@i function mean(out!, x)
    anc ← zero(out!)
    for i=1:length(x)
        anc += x[i]
    end
    mulint(out!, length(x))
end

"""
    var_and_mean_sq(var!, mean!, sqv)

the variance and mean value from squared values.
"""
@i function var_and_mean_sq(var!, mean!, sqv::AbstractVector{T}) where T
    @zeros T mean_sum var_sum
    @inbounds for i=1:length(sqv)
        mean_sum += sqv[i] ^ 0.5
        var_sum += sqv[i]
    end
    mean! += mean_sum / length(sqv)
    @routine begin
        @zeros T var_anc1 var_anc2 sqmean
        var_anc1 += var_sum / length(sqv)
        sqmean += mean! ^ 2
        var_anc1 -= sqmean
        var_anc2 += var_anc1 * length(sqv)
    end
    var! += var_anc2 / (length(sqv)-1)
    ~@routine
    PUSH!(var_sum)
    PUSH!(mean_sum)
end

"""
Squared distance of two vertices.
"""
@i @inline function sqdistance(dist!, x1::AbstractVector{T}, x2::AbstractVector) where T
    @invcheckoff @inbounds for i=1:length(x1)
        x1[i] -= x2[i]
        dist! += x1[i] ^ 2
        x1[i] += x2[i]
    end
end

"""The loss of graph embedding problem."""
@i function embedding_loss(out!::T, x) where T
    @zeros T v1 m1 v2 m2 diff
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
        m1 -= m2 - 0.1
    end
    out! += v1 + v2
    if (m1 > 0, ~)
        # to ensure mean(v2) > mean(v1)
        # if mean(v1)+0.1 - mean(v2) > 0, punish it.
        out! += exp(m1)
        out! -= 1
    end
    ~@routine
end
