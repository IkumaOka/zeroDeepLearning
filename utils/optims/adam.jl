mutable struct Adam
    Adam() = new()
    lr::Float64
    beta1::Float64
    beta2::Float64
    iter::Int64
    m
    v
end

function Adam(lr, beta1, beta2)
    self = Adam()
    self.lr = lr
    self.beta1 = beta1
    self.beta2 = beta2
    self.iter = 0
    self.m = nothing
    self.v = nothing
    return self
end

function update(self::Adam, params, grads)
    if self.m == nothing
        self.m = Dict{}()
        self.v = Dict{}()
        for (key, value) in params
            self.m[key] = zeros(size(value))
            self.v[key] = zeros(size(value))
        end
    end

    self.iter += 1
    lrt = self.lr * sqrt(1.0 - self.beta2 ^ self.iter) / (1.0 - self.beta1 ^ self.iter)

    for key in keys(params)
        self.m[key] += (1 - self.beta1) .* (grads[key] - self.m[key])
        self.v[key] += (1 - self.beta2) .* (grads[key] .^ 2 - self.v[key])
        
        params[key] -= lrt .* self.m[key] ./ (sqrt.(self.v[key]) .+ 1e-7)
    end
end

