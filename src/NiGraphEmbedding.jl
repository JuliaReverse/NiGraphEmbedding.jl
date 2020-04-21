module NiGraphEmbedding

using NiLang, NiLang.AD
import ForwardDiff
using ForwardDiff: Dual
using Optim

export embedding_loss

include("native.jl")
include("reversible.jl")
include("hessian.jl")
include("Adam.jl")
include("train.jl")

end # module
