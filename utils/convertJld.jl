# pickleデータをJLD形式に変換する

using PyCall, JLD2
py"""
import pickle

with open("../params/sample_weight.pkl", 'rb') as f:
    data = pickle.load(f)
    print(data["W1"].shape)
    print(data["W1"])
    # print(len(data["W1"])) # 784
"""
data = py"data"
println("-------------------------------------------------------------------")
data_W1 = reshape(data["W1"], 784, 50)
println(data_W1 == data["W1"])
println(typeof(data["W1"]))
# println(data["W1"][1, :])
# println(length(data["W1"][1, :]))
# println(length(data["W1"][:, 1]))
save("sampleWeight.jld", data)