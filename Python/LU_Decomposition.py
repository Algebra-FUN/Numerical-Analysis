import numpy as np

def LU(A:np.ndarray):
    assert A.ndim == 2
    n = A.shape[0]
    assert n == A.shape[1]
    L = np.eye(n,n)
    U = np.zeros((n,n))
    for r in range(n):
        U[r,r:n] = A[r,r:n] - L[r,0:r]@U[0:r,r:n]
        L[r+1:n,r] = (A[r+1:n,r] -L[r+1:n,0:r]@U[0:r,r])/U[r,r]
    return L,U

if __name__ == '__main__':
    A = np.array([
        [1,2,3],
        [2,5,2],
        [3,1,5]
    ])
    print(LU(A))