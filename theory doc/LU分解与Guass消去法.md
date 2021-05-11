## LU分解与列主元的Gauss消去法

### LU分解

直接三角分解的计算公式

计算$U$的第$r$行,$L$的第$r$列元素($r=1,2,\cdots,n$)
$$
u_{ri}=a_{ri}-\sum_{k=1}^{r-1}l_{rk}u_{ki}\qquad i=r,r+1,\cdots,n;\\
l_{ir}=(a_{ir}-\sum_{k=1}^{r-1}l_{ik}u_{kr})/u_{rr}\qquad i=r,r+1,\cdots,n,且r\neq n
$$
求解$Ly=b,Ux=y$的计算公式：
$$
\begin{cases}
y_1=b_1,\\
y_i=b_i-\sum_{k=1}^{i-1}l_{ik}y_k \qquad i=2,3,\cdots,n
\end{cases}
$$

$$
\begin{cases}
x_n=y_n/u_{nn},\\
x_i=(y_i-\sum_{k=i+1}^n u_{ik}x_k)/u_{ii} \qquad i=2,3,\cdots,n
\end{cases}
$$

### 列主元的Gauss消去法

1. 对于$i=1,2,\cdots,n-1$

    1. 选取列主元
       $$
       |a_{i_m,i}|=\max_{i\leq k\leq n}|a_{ki}|
       $$

    2. 若列主元$a_{i_m,i}=0$，则停止

    3. 若$iₘ \neq i$,则交换行
       $$
       a_{ij}\leftrightarrow a_{i_m,j}\\
       b_i\leftrightarrow b_{i_m}
       $$

    4. 消元计算

       对于$k=i+1,\cdots,n$
       $$
       a_{kj}\leftarrow a_{kj}-\frac{a_{ki}}{a_{ii}} a_{ij}\\
       b_{k}\leftarrow b_k - \frac{a_{ki}}{a_{ii}}b_i
       $$

2. 回代求解
	1. $b_n\leftarrow b_n/a_{nn}$
	2. 对于$i=n-1,\cdots,2,1$
		$$
		b_i\leftarrow (b_i-\sum_{j=i-1}^n a_{ji}b_j)/a_{ii}
		$$

最终$x\leftarrow b$

|          | 直接$LU$分解法 | 列主元的$Gauss$消去法 |
| :------- | :------------: | :-------------------: |
| 误差$δb$ | $10e{-9}$量级  |    $10e{-16}$量级     |
| 性能     |    2.908 μs    |       2.655 μs        |

