# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Julia
#     language: julia
#     name: julia-1.6
# ---

# <center><h1>LU分解与列主元的Gauss消去法解线性方程组<h1/><center/>

# ## 实验内容
# 用LU分解与列主元的Gauss消去法解线性方程组
#
# $$
# \begin{pmatrix}10.0&-7.0&0.0&1.0\\-3.0&2.099999&6.0&2.0\\5.0&-1.0&5.0&-1.0\\2.0&1.0&0.0&2.0\end{pmatrix}
# \begin{pmatrix}x_1\\x_2\\x_3\\x_4\end{pmatrix}=\begin{pmatrix}8.0\\5.900001\\5.0\\1.0\end{pmatrix}
# $$
#
# (1)用$A=LU$分解，求$L,U$,并求解向量$x$
#
# (2)用列主元的Gauss消去法求解向量$x$

# ## 实验原理
#
# ### LU分解
#
# 直接三角分解的计算公式
#
# 计算$U$的第$r$行,$L$的第$r$列元素($r=1,2,\cdots,n$)
# $$
# u_{ri}=a_{ri}-\sum_{k=1}^{r-1}l_{rk}u_{ki}\qquad i=r,r+1,\cdots,n;\\
# l_{ir}=(a_{ir}-\sum_{k=1}^{r-1}l_{ik}u_{kr})/u_{rr}\qquad i=r,r+1,\cdots,n,且r\neq n
# $$
# 求解$Ly=b,Ux=y$的计算公式：
# $$
# \begin{cases}
# y_1=b_1,\\
# y_i=b_i-\sum_{k=1}^{i-1}l_{ik}y_k \qquad i=2,3,\cdots,n
# \end{cases}
# $$
#
# $$
# \begin{cases}
# x_n=y_n/u_{nn},\\
# x_i=(y_i-\sum_{k=i+1}^n u_{ik}x_k)/u_{ii} \qquad i=2,3,\cdots,n
# \end{cases}
# $$

# ### 列主元的Gauss消去法
#
# 1. 对于$i=1,2,\cdots,n-1$
#
#     1. 选取列主元
#        $$
#        |a_{i_m,i}|=\max_{i\leq k\leq n}|a_{ki}|
#        $$
#
#     2. 若列主元$a_{i_m,i}=0$，则停止
#
#     3. 若$iₘ \neq i$,则交换行
#        $$
#        a_{ij}\leftrightarrow a_{i_m,j}\\
#        b_i\leftrightarrow b_{i_m}
#        $$
#
#     4. 消元计算
#
#        对于$k=i+1,\cdots,n$
#        $$
#        a_{kj}\leftarrow a_{kj}-\frac{a_{ki}}{a_{ii}} a_{ij}\\
#        b_{k}\leftarrow b_k - \frac{a_{ki}}{a_{ii}}b_i
#        $$
#
# 2. 回代求解
#     1. $b_n\leftarrow b_n/a_{nn}$
#     2. 对于$i=n-1,\cdots,2,1$
#         $$
#         b_i\leftarrow (b_i-\sum_{j=i-1}^n a_{ji}b_j)/a_{ii}
#         $$
#
# 最终$x\leftarrow b$

# ## 编程实现、计算实例、数据、结果
# > 用Julia语言实现

# 先导入依赖和输入计算所需的数据

# +
using LinearAlgebra,BenchmarkTools,LambdaFn

include("MatrixKit.jl")
include("Swap.jl");
# -

@latex A = [10 -7 0 1;-3 2.099999 6 2;5 -1 5 -1;2 1 0 2]

@latex b = [8;5.900001;5;1]

# ### LU分解方法计算

function LU(A::Matrix)
    n = size(A)[1]
    @assert n == size(A)[2] "The Matrix must be square matrix!"
    L = eye(n)
    U = zeros(n,n)
    for r = 1:n
        U[r:r,r:n] = A[r:r,r:n] - L[r:r,1:r-1]*U[1:r-1,r:n]
        L[r+1:n,r:r] = (A[r+1:n,r:r] - L[r+1:n,1:r-1]*U[1:r-1,r:r])/U[r,r]
    end
    return L,U
end;

function LinearSolveWithLU(L::Matrix,U::Matrix,b::Vector)
    n = length(b)
    y = zeros(n)
    for i = 1:n
        y[i:i,1:1] = b[i:i,1:1] - L[i:i,1:i-1]*y[1:i-1,1:1]
    end
    x = zeros(n)
    for i = n:-1:1
        x[i:i,1:1] = (y[i:i,1:1]-U[i:i,i+1:n]*x[i+1:n,1])/U[i,i]
    end
    return x
end;

# #### 计算结果

L,U=LU(A);
@latex L

@latex U

detA = (prod∘diag)(U)

@latex x=LinearSolveWithLU(L,U,b)

# #### 性能

@benchmark LinearSolveWithLU(L,U,b)

# #### 误差比较
#
# $$
# \delta A = LU -A
# $$
#
# 其中$L,U$为计算结果

@latex δA = L*U - A

# > 可见误差还是比较小

# $$\delta b = Ax-b$$
#
# 其中$x$为计算结果

@latex δb = A*x - b

# 误差$δb$在$10e-9$量级，还是比较小

# ### 列主元的Gauss消去法

function GaussEliminate(A::Matrix,b::Vector)
    n = length(b)
    Ab = [A b]
    for i = 1:n-1
        iₘ = i-1+argmax(abs.(Ab[i:n,i]))
        iₘ == i || @swap Ab[i,i:end],Ab[iₘ,i:end]
        Ab[i+1:n,i+1:end] -= Ab[i+1:n,i:i]*Ab[i:i,i+1:end]/Ab[i,i]
    end
    x = Ab[:,n+1]
    for i = n:-1:1
        x[i] /= Ab[i,i]
        x[1:i-1] -= Ab[1:i-1,i]*x[i]
    end
    return x
end;

# #### 计算结果

@latex x = GaussEliminate(A,b)

# #### 性能

@benchmark GaussEliminate(A,b)

# #### 误差比较
#
# $$\delta b = Ax-b$$
#
# 其中$x$为计算结果

@latex δb = A*x - b

# 误差$δb$在$10e-16$量级，还是比较小

# ### *列主元的LU分解*
# > 补充内容

function P⁻¹LU(A::Matrix)
    # deepcopy otherwise you will change the origin data
    A = A |> deepcopy
    n = size(A)[1]
    @assert n == size(A)[2] "The Matrix must be square matrix!"
    Iₚ = zeros(Int8,n)
    for r = 1:n
        #1 calculate s
        A[r:n,r:r] -= A[r:n,1:r-1]*A[1:r-1,r:r]
        
        #2 choose main element
        iᵣ = r-1+argmax(abs.(A[r:n,r]))
        Iₚ[r] = iᵣ
        
        #3 swap row of A
        r == iᵣ || @swap A[r,:],A[iᵣ,:]
        
        #4 calculate L,U same as function LU
        A[r+1:n,r:r] /= A[r,r]
        A[r:r,r+1:n] -= A[r:r,1:r-1]*A[1:r-1,r+1:n]
    end
    return A,Iₚ
end;

function LinearSolveWithP⁻¹LU(LUᵤₙᵢₒₙ ::Matrix,Iₚ::Vector,b::Vector)
    b = b |> copy
    n = length(Iₚ)
    for i = 1:n-1
        i == Iₚ[i] || @swap b[i],b[Iₚ[i]]
    end
    for i = 2:n
        b[i] -= LUᵤₙᵢₒₙ[i,1:i-1]'*b[1:i-1]
    end
    b[n] /= LUᵤₙᵢₒₙ[n,n]
    for i = n-1:-1:1
        b[i] -= LUᵤₙᵢₒₙ[i,i+1:n]'*b[i+1:n]
        b[i] /= LUᵤₙᵢₒₙ[i,i]
    end
    return b
end;

LUᵤₙᵢₒₙ,Iₚ=P⁻¹LU(A);
@latex LUᵤₙᵢₒₙ

@latex Iₚ T

@latex x = LinearSolveWithP⁻¹LU(LUᵤₙᵢₒₙ,Iₚ,b)

# ## 结果分析与比较
# 根据上面的计算结果
#
# |          | 直接$LU$分解法 | 列主元的$Gauss$消去法 |
# | :--------: | :--------------: | :---------------------: |
# | 误差$δb$ | $10e{-9}$量级  | $10e{-16}$量级        |
# | 性能     | 2.908 μs       | 2.655 μs              |
#
# > 这里性能采用运行`中位数时间(median time)`度量
#
# ### 分析结论:
#
# 计算精确度：列主元的$Gauss$消去法比直接$LU$分解更高
#
# 性能：列主元的$Gauss$消去法比直接$LU$分解更好
#
# > 这与理论分析的结论一致
