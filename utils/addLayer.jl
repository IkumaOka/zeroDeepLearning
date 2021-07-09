mutable struct AddLayer
    AddLayer() = new()
    x
    y
end

function AddLayer(apple)
    self = AddLayer()
    self.x = nothing
    self.y = nothing
    return self
end

function forward(self::AddLayer, x, y)
    out = x + y

    return out
end

function backward(self::AddLayer, dout)
    dx = dout * 1
    dy = dout * 1

    return dx, dy
end