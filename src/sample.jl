function p(x::Array{Int64,2}, y::Array{Int64,2})
    return x .+ y
end

x = [1 2 3]
#x = 1
y = [1 2 3]
a = p(x, y)
println(a)