mutable struct MulLayer
    MulLayer() = new()
    x
    y
end

function MulLayer(apple)
    self = MulLayer()
    self.x = nothing
    self.y = nothing
    return self
end

function forward(self::MulLayer, x, y)
    self.x = x
    self.y = y
    out = x * y

    return out
end

function backward(self, dout)
    dx = dout * self.y # xとyをひっくり返す
    dy = dout * self.x

    return dx, dy
end