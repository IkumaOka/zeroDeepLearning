mutable struct SGD
    SGD() = new()
    lr::Float64
end

function SGD(lr)
    self = SGD()
    self.lr = lr
    return self
end

function update(self::SGD, params, grads)
    # p.167の式(6.1)を実装
    for key in keys(params)
        params[key] -= self.lr .* grads[key]
    end
end

