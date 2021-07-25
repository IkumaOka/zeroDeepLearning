mutable struct Momentum
    Momentum() = new()
    lr::Float64
    momentum::Float64
    v
end

function Momentum(lr, momentum)
    self = Momentum()
    self.lr = lr
    self.momentum = momentum
    self.v = nothing
    return self
end

function update(self::Momentum, params, grads)
    if self.v == nothing
        self.v = Dict{}()
        for (key, value) in params
            self.v[key] = zeros(value)
        end
        for key in keys(params)
            # p.170の式(6.3), (6.4)を実装
            self.v[key] = self.momentum .* self.v[key] - self.lr .* grads[key]
            params[key] += self.v[key]
        end
    end
end

