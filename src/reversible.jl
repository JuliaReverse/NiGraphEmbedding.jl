"""
Squared distance of two vertices.
"""
@i @inline function sqdistance(dist!, x1::AbstractVector{T}, x2::AbstractVector) where T
    @invcheckoff @inbounds for i=1:length(x1)
        @routine begin
            d ← zero(T)
            d += x1[i] - x2[i]
        end
        dist! += d^2
        ~@routine
    end
end

"""
    var_and_mean_sq(var!, mean!, sqv)

the variance and mean value from squared values.
"""
@i function var_and_mean_sq(var!, var_sum, mean!, mean_sum, sqv::AbstractVector{T}) where T
    @inbounds for i=1:length(sqv)
        mean_sum += sqv[i] ^ 0.5
        var_sum += sqv[i]
    end
    mean! += mean_sum / (@const length(sqv))
    @routine begin
        @zeros T var_anc1 var_anc2 sqmean
        var_anc1 += var_sum / (@const length(sqv))
        sqmean += mean! ^ 2
        var_anc1 -= sqmean
        var_anc2 += var_anc1 * (@const length(sqv))
    end
    var! += var_anc2 / (@const length(sqv)-1)
    ~@routine
end

"""The loss of graph embedding problem."""
@i function embedding_loss(out!::T, x) where T
    @routine @invcheckoff begin
        d1 ← zeros(T, length(L1))
        d2 ← zeros(T, length(L2))
        for i=1:length(L1)
            @inbounds sqdistance(d1[i], x[:,L1[i][1]],x[:,L1[i][2]])
        end
        for i=1:length(L2)
            @inbounds sqdistance(d2[i], x[:,L2[i][1]],x[:,L2[i][2]])
        end
        @zeros T v1 m1 v2 m2 vacc1 vacc2 acc1 acc2
        var_and_mean_sq(v1, vacc1, m1, acc1, d1)
        var_and_mean_sq(v2, vacc2, m2, acc2, d2)
        m1 -= m2 - 0.1
    end
    out! += v1 + v2
    if m1 > 0
        # to ensure mean(v2) > mean(v1)
        # if mean(v1)+0.1 - mean(v2) > 0, punish it.
        out! += exp(m1)
        out! -= 1
    end
    ~@routine
end
