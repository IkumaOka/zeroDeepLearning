mutable struct ReLULayer
    # いらないかも。Python実装はあるから一応残す。(p.142)
    mask
end

function forward(self::ReLULayer, x)
    out = copy(x)
    return ReLUActivation.(out)
end

function backward(self::ReLULayer, dout)
    out = copy(dout)
    return ReLUActivation.(out)
end

```
0以下で0、それ以外はその値を返す関数。
この関数自体は、スカラーに対して適用する関数なので、
forward関数で使っているように「.」を使ってブロードキャストする。
```
function ReLUActivation(x::Float64)
    if x <= 0
        x = 0
    end
    return x
end