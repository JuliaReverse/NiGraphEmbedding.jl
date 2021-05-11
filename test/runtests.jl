using NiGraphEmbedding
import LinearAlgebra, Statistics
using Test, Random
using NiLang, NiLang.AD
using Zygote
using Optim

@testset "gradient" begin
    Random.seed!(2)
    params = randn(5, 10)
    x = randn(5,10)
    @test embedding_loss(0.0, x)[1] ≈ embedding_loss(x)
    @test grad(Grad(embedding_loss)(Val(1), 0.0, x)[3]) ≈ Zygote.gradient(embedding_loss, x)[1]
end

@testset "train" begin
    Random.seed!(2)
    params = randn(5, 10)
    train(params, Adam(lr=0.01))
    # distances of connected bonds.
    d1s = [LinearAlgebra.norm(params[:,i]-params[:,j]) for (i,j) in NiGraphEmbedding.L1]
    # distances of disconnected bonds.
    d2s = [LinearAlgebra.norm(params[:,i]-params[:,j]) for (i,j) in NiGraphEmbedding.L2]
    @test abs(Statistics.var(d1s)) < 1e-5
    @test abs(Statistics.var(d2s)) < 1e-5

    params = randn(5, 10)
    opt = NewtonTrustRegion(; initial_delta = 1.0,
                    delta_hat = 20.0,
                    eta = 0.1,
                    rho_lower = 0.25,
                    rho_upper = 0.75)
    train(params, opt)

    d1s = [LinearAlgebra.norm(params[:,i]-params[:,j]) for (i,j) in NiGraphEmbedding.L1]
    # distances of disconnected bonds.
    d2s = [LinearAlgebra.norm(params[:,i]-params[:,j]) for (i,j) in NiGraphEmbedding.L2]
    @test abs(Statistics.var(d1s)) < 1e-5
    @test abs(Statistics.var(d2s)) < 1e-5
end
