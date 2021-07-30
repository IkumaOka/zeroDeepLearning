mutable struct AdaGrad
    AdaGrad() = new()
    lr::Float64
    h
end

function AdaGrad(lr)
    self = AdaGrad()
    self.lr = lr
    self.h = nothing
    return self
end

function update(self::AdaGrad, params, grads)
    if self.h == nothing
        self.h = Dict{}()
        for (key, value) in params
            if typeof(value) == Float64
                self.h[key] = 0.0
            else
                self.h[key] = zeros(size(value))
            end
        end
    end
    # p.167の式(6.1)を実装
    for key in keys(params)
        # p.173の式(6.5), (6.6)を実装
        self.h[key] += grads[key] .* grads[key]
        # 要素同士の割り算をしたいときは「./」を使う。「/」を使うと行列演算になる。
        params[key] -= self.lr .* grads[key] ./ (sqrt.(self.h[key]) .+ 1e-7)
    end
end

