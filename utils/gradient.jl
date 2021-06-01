# 数値微分
function numericalDiff(f, x)
    h = 1e-4
    return (f(x+h) - f(x - h)) / (2 * h)
end