# ミニバッチ配列を返す（現状Numpyのように簡単にミニバッチを取得する方法がわからない）
function miniBatch(batchMask, array)
    miniBatchArray = []
    for i in 1:length(batchMask)
        push!(miniBatchArray, array[batchMask[i], :])
    end
    return hcat(miniBatchArray...)'
end