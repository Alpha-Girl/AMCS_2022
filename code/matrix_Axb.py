import numpy as np
A = np.array([[1, 5, 3, 1],
            [4, 2, 6, 3],
            [1, 4, 3, 2],
            [4, 4, 1, 1],
            [5,5,2,3]
            ])
print(np.dot(A.T,A))
# b = np.array([2,1,2,5,0])
# X=np.linalg.solve(A,b)
# mat = 1/4*np.dot(A,A.T)
mat = 1/5*np.dot(A.T,A)

eigenvalue, featurevector = np.linalg.eig(mat)

print("特征值：", eigenvalue)
print("特征向量：", featurevector)
B=np.array([[ 0.50747081,  0.27937059 , 0.80877607, -0.10152201],
 [ 0.33633553, -0.91389257,  0.12820927 , 0.18772631]])
mat_2=np.dot(B,A.T)
print(mat_2)