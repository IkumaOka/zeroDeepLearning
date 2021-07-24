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
    for key in keys(params)
        params[key] -= self.lr .* grads[key]
    end
end

