using BenchmarkTools
using NiGraphEmbedding
using ForwardDiff, NiLang, Zygote, ReverseDiff

function get_hessian_reversediff(params0::AbstractArray{T}) where T
    N = length(params0)
    params = Dual.(params0, zero(T))
    hes = zeros(T, N, N)
    for i=1:N
        @inbounds i !== 1 && (params[i-1] = Dual(params0[i-1], zero(T)))
        @inbounds params[i] = Dual(params0[i], one(T))
        res = ReverseDiff.gradient(embedding_loss, params)
        hes[:,i] .= vec(ForwardDiff.partials.(res, 1))
    end
    hes
end

suite = BenchmarkGroup()
suite["NiLang"] = BenchmarkGroup(["Term"])
suite["ForwardDiff"] = BenchmarkGroup(["Term"])
suite["Zygote"] = BenchmarkGroup(["Term"])
suite["ReverseDiff"] = BenchmarkGroup(["Term"])

cases = [("NiLang", "Call"), ("NiLang", "Uncall"),
         ("NiLang", "Gradient"), ("NiLang", "Hessian"),
         ("ForwardDiff", "Call"), ("ForwardDiff", "Gradient"),
         ("ForwardDiff", "Hessian"),
         ("Zygote", "Gradient"),
         ("ReverseDiff", "Gradient"), ("ReverseDiff", "Hessian")
         ]

for (lang, term) in cases
    suite[lang][term] = BenchmarkGroup(["dimension"])
end

for k=1:10
    suite["NiLang"]["Call"][k] = @benchmarkable embedding_loss(0.0, $(randn(k, 10)))
    suite["NiLang"]["Uncall"][k] = @benchmarkable (~embedding_loss)(0.0, $(randn(k, 10)))
    suite["NiLang"]["Gradient"][k] = @benchmarkable NiLang.AD.gradient(Val(1), embedding_loss, (0.0, $(randn(k, 10))))
    suite["NiLang"]["Hessian"][k] = @benchmarkable get_hessian($(randn(k, 10)))
    suite["ForwardDiff"]["Call"][k] = @benchmarkable embedding_loss($(randn(k, 10)))
    suite["ForwardDiff"]["Gradient"][k] = @benchmarkable ForwardDiff.gradient(embedding_loss, $(randn(k, 10)))
    suite["ForwardDiff"]["Hessian"][k] = @benchmarkable ForwardDiff.hessian(embedding_loss, $(randn(k, 10)))
    suite["Zygote"]["Gradient"][k] = @benchmarkable Zygote.gradient(embedding_loss, $(randn(k, 10)))
    suite["ReverseDiff"]["Gradient"][k] = @benchmarkable ReverseDiff.gradient!(ctape, $((randn(k, 10),))) setup=(ctape = ReverseDiff.compile(ReverseDiff.GradientTape(embedding_loss, (randn(k, 10),))))
    suite["ReverseDiff"]["Hessian"][k] = @benchmarkable get_hessian_reversediff($(randn(k, 10)))
    #suite["ReverseDiff"]["Hessian"][k] = @benchmarkable ReverseDiff.hessian(embedding_loss, $(randn(k, 10)))
end

tune!(suite)
res = run(suite)#; seconds=100, samples=1000)

function analyze_res(res)
    times = zeros(10, length(cases))
    for (k, (lang, term)) in enumerate(cases)
        for i=1:10
            @show lang, term
            times[i,k] = minimum(res[lang][term][i].times)
        end
    end
    return times
end

times = analyze_res(res)
using DelimitedFiles
writedlm(joinpath(dirname(@__FILE__), "bench_graphembedding.dat"), times)
