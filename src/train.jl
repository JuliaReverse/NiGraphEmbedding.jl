export train

function train(params, opt::Adam; maxiter=2000)
    # mask used to fix first two elements
    msk = [false, false, true, true, true, true, true, true, true, true]
    pp = params[:,msk]
    for i=1:maxiter
        g = NiLang.AD.gradient(Val(1), embedding_loss, (0.0, params))[2][:,msk]
        update!(pp, g, opt)
        view(params, :, msk) .= pp
        if i%1000 == 0
            println("Step $i, loss = $(embedding_loss(0.0, params)[1])")
        end
    end
    params
end

using Optim
function train(params, opt::NewtonTrustRegion)
    # mask used to fix first two elements
    msk = [false, false, true, true, true, true, true, true, true, true]
    i = Ref(1)
    function f(x)
        vec(view(params,:,msk)) .= x
        l = embedding_loss(0.0, params)[1]
        println("Step $(i[]), loss = $l")
        return l
    end
    function g!(G, x)
        vec(view(params,:,msk)) .= x
        G .= vec(NiLang.AD.gradient(Val(1), embedding_loss, (0.0, params))[2][:,msk])
    end
    function h!(H, x)
        vec(view(params,:,msk)) .= x
        nm = sum(msk)*size(params, 1)
        i[] += 1
        H .= reshape(reshape(get_hessian(params), size(params)..., size(params)...)[:,msk,:,msk], nm, nm)
    end
    optimize(f, g!, h!, vec(params[:,msk]), opt)
    params
end
