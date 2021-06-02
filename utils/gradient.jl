# 1変数の数値微分
function numericalDiff(f, x)
    h = 1e-4
    return (f(x+h) - f(x - h)) / (2 * h)
end

# 2変数以上の数値微分
function numericalGradient(f, x)
    h = 1e-4
    grad = zeros(size(x))
    for idx in 1:ndims(x)
        tmpVal = x[idx]
        #f(x+h)の計算
        x[idx] = tmpVal .+ h
        fxh1 = f(x)

        #f(x-h)の計算
        x[idx] = tmpVal .- h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) ./ (2 * h)
        #値を元に戻す
        x[idx] = tmpVal
    end
    return grad
end

# 勾配降下法
function gradientDescent(f, initX, lr=0.01, stepNum=100)
    x = initX
    for i in 1:stepNum
        grad = numericalGradient(f, x)
        x .-= lr .* grad
    end

    return x
end