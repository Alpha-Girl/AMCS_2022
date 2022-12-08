import numpy as np
C = np.array([[1, 5, 3, 1],
            [4, 2, 6, 3],
            [1, 4, 3, 2],
            [4, 4, 1, 1],
            [5,5,2,3]
            ])
A = np.array([[-2, 1, 0, -1],
            [1, -2, 3, 1],
            [-2, 0, 0, 0],
            [1, 0, -2, -1],
            [2,1,-1,1]
            ])
print(np.dot(A.T,A))
# b = np.array([2,1,2,5,0])
# X=np.linalg.solve(A,b)
# mat = 1/4*np.dot(A,A.T)
mat = 1/5*np.dot(A.T,A)

eigenvalue, featurevector = np.linalg.eig(mat)

print("特征值：", eigenvalue)
print("特征向量：", featurevector)
B=np.array([[ -0.26087912,  0.91114239, -0.30581868, -0.09075555],
 [ 0.4801165 ,  0.0378551  ,-0.5085857  , 0.71371964]])
mat_2=np.dot(B,C.T)
print(mat_2)