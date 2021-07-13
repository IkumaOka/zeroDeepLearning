mutable struct ReluLayer
    # いらないかも。Python実装はあるから一応残す。(p.142)
    mask
end

function forward(self::ReluLayer, x)
    out = copy(x)
    return reluActivation.(out)
end

function backward(self::ReluLayer, dout)
    out = copy(dout)
    return reluActivation.(out)
end

```
0以下で0、それ以外はその値を返す関数。
この関数自体は、スカラーに対して適用する関数なので、
forward関数で使っているように「.」を使ってブロードキャストする。
```
function reluActivation(x::Float64)
    if x <= 0
        x = 0
    end
    return x
end