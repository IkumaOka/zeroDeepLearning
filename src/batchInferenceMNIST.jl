include("./learnedMNISTparams.jl")

# shape of x = 28 * 28 * 10000
x, t = getData()
network = initNetwork()

batchSize = 100
accuracyCnt = 0

for i in 1:100:(length(x[28, 28, :]) + 1 - batchSize)
    Xbatch = x[:, :, i:i+batchSize-1]
    Xi = reshape(Xbatch, length(Xbatch[:, :, 1]), batchSize)
    yBatch = predict(network, Xi')
    p = mapslices(argmax, yBatch, dims=2)
    println(p .== t[i:i+batchSize-1])
    global accuracyCnt += sum(p .== t[i:i+batchSize-1])
end
print("Accuracy:", accuracyCnt / length(x[28, 28, :]))