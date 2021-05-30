include("./learnedMNISTparams.jl")

# shape of x = 28 * 28 * 10000
x, t = getData()
network = initNetwork()

accuracyCnt = 0
for i in 1:length(x[28, 28, :])
    Xi =  reshape(x[:, :, i]', length(x[:, :, i]), 1)
    y = predict(network, Xi')
    # println(y)
    p = findmax(y)[2][2]
    # println(p)
    if p == t[i]
        global accuracyCnt += 1
    end
end
print("Accuracy:", accuracyCnt / length(x[28, 28, :]))