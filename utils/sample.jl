# pickleデータをJLD形式に変換する
using PyCall
py"""
import numpy as np
data = np.array([[1, 2, 3], [4, 5, 6]])
"""
data = py"data"
println("-------------------------------------------------------------------")
println(typeof(data))
# println(data["W1"][1, :])
# println(length(data["W1"][1, :]))
# println(length(data["W1"][:, 1]))