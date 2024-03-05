from distutils import debug
from flask import Flask, render_template, request, jsonify,url_for
import math
import numpy as np

app=Flask(__name__,static_folder='static')

def est_triangulaire_superieure(matrice):
    n = len(matrice)
    for i in range(n):
        for j in range(i):
            if matrice[i][j] != 0:
                return False
    return True

def est_triangulaire_inferieure(matrice):
    n = len(matrice)
    for i in range(n):
        for j in range(i + 1, n):
            if matrice[i][j] != 0:
                return False
    return True

def transopse(matrice,n):
    transpose=[[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            transpose[i][j]=matrice[j][i]
    return transpose

def symetrique(matrice):
    n = len(matrice)
    for i in range(n):
        for j in range(i + 1, n):
            if matrice[i][j] != matrice[j][i]:
                return False
    return True

def res_sup_dense(matrice, b, n):

    if not est_triangulaire_superieure(matrice):
                raise ValueError("Attention!\nVotre matrice n'est pas une matrice triangulaire supérieure")


    x = [0] * n
    for i in range(n - 1, -1, -1):
        if matrice[i][i] == 0:
            raise ValueError("")

        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= matrice[i][j] * x[j]
        x[i] /= matrice[i][i]
    return x

def res_sup_densee(matrice, b, n):

    x = [0] * n
    for i in range(n - 1, -1, -1):
        if matrice[i][i] == 0:
            raise ValueError("")

        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= matrice[i][j] * x[j]
        x[i] /= matrice[i][i]
    return x

def res_inf_dense(matrice, b, n):
    if not est_triangulaire_inferieure(matrice):
                raise ValueError("Attention!\nVotre matrice n'est pas une matrice triangulaire inférieure")
    x = [0] * n
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] = x[i] - matrice[i][j] * x[j]
        x[i] = x[i] / matrice[i][i]
    return x

def res_inf_densee(matrice, b, n):
    x = [0] * n
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] = x[i] - matrice[i][j] * x[j]
        x[i] = x[i] / matrice[i][i]
    return x

def matrice_demi_bande_sup(matrice,n,largeur):
    m= largeur-1
    for i in range(n):
        for j in range(n):
            if (i>j or j-i>m) and matrice[i][j]!=0:
                return False
    return True

def res_demi_bande_sup(matrice, b, n, largeur):
    if not matrice_demi_bande_sup(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice demi-bande supérieure")
    m=largeur-1
    x = [0] * n
    for i in range(n - 1, -1, -1):
        if matrice[i][i] == 0:
            raise ValueError("Le diagonal ne doit pas etre nul")

        x[i] = b[i]
        for j in range(i + 1, min(i + m + 1, n)):
            x[i] -= matrice[i][j] * x[j]
        x[i] /= matrice[i][i]
    return x

def res_demi_bande_inf(matrice, b, n, largeur):
    if not matrice_demi_bande_inf(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice demi-bande inférieure")
    m=largeur-1
    x = [0] * n
    for i in range(n):
        if matrice[i][i] == 0:
            raise ValueError("Le diagonal ne doit pas etre nul")

        x[i] = b[i]
        for j in range(max(0, i - m), i):
            x[i] -= matrice[i][j] * x[j]
        x[i] /= matrice[i][i]
    return x

def pivot(matrice, k, n):
    indice_max = k
    valeur_max = abs(matrice[k][k])
    for i in range(k + 1, n):
        if abs(matrice[i][k]) > valeur_max:
            valeur_max = abs(matrice[i][k])
            indice_max = i
    return indice_max


def matrice_bande(matrice,n,largeur):
    m=(largeur-1)//2 
    if (m>(n-1)/2):
        raise ValueError("Attention!\nLa largeur bande spécifiée est trop grande pour cette matrice")
        
    for i in range(n):
        for j in range(n):
            if abs(i-j)>m and matrice[i][j]!=0:
                return False
                
    return True

def matrice_demi_bande_inf(matrice,n,largeur):
    m= largeur-1
    for i in range(n):
        for j in range(n):
            if (i<j or i-j>m) and matrice[i][j]!=0:
                return False
    return True


def LU_dense(matrice,b,n):
    if(symetrique(matrice)==False):
        raise ValueError("Attention!\nVotre matrice doit être symétrique")
    '''print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))'''

    for k in range(n):
       if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")

       ''' if determinant(decomposition_sous_matrice(matrice, k+1)) == 0:
            raise ValueError("Attention!\nVotre matrice n'admet pas une décomposition LU")'''
            
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):

        L[i][i] = 1

        for j in range(i, n):
            somme_u = 0

            for k in range(i):
                somme_u+=U[k][j]*L[i][k]

            U[i][j] = matrice[i][j] - somme_u

        for j in range(i+1, n):
            L[j][i] = (U[i][j] ) / U[i][i]
    y=res_inf_densee(L,b,n) #Méthode descente
    x=res_sup_densee(U,y,n) #Méthode remontée

    return {'solution': x, 'L': L, 'U': U}

def LU_bande(matrice,n,b,largeur):
    if not matrice_bande(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
    if(symetrique(matrice)==False):
        raise ValueError("Attention!\nVotre matrice doit être symétrique")
    '''print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))'''

    for k in range(n):           
        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")
            
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1

        for j in range(i,min(i+largeur, n)):
            somme_u = 0
            for k in range(max(i-largeur+1, 0),i):
                somme_u += U[k][j] * L[i][k]
            U[i][j] = matrice[i][j] - somme_u

        for j in range(i+1,min(i+largeur,n)):
            L[j][i] = U[i][j] / U[i][i]

    y=res_inf_densee(L,b,n) #Méthode descente
    x=res_sup_densee(U,y,n) #Méthode remontée
    print("La solution X=")

    return {'solution': x, 'L': L, 'U': U}

def Gauss_pivotage_dense(matrice,b,n):

    if(symetrique(matrice)==True):
        raise ValueError("Attention!\nVotre matrice ne doit pas être symétrique")
    for k in range(n-1):
        pivot_indice = pivot(matrice, k, n)
        if pivot_indice != k:
            echanger_lignes(matrice, b, k, pivot_indice)
        for i in range(k+1,n):
            matrice[i][k]=matrice[i][k]/matrice[k][k]
            for j in range(k+1,n):
                matrice[i][j] -= matrice[i][k]*matrice[k][j]
            b[i] -= matrice[i][k] * b[k]

    x=res_sup_densee(matrice,b,n)
    return x

def Gauss_pivotage_bande(matrice,b,n,largeur):
    m=(largeur-1)//2
    if not matrice_bande(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")

    if symetrique(matrice)==True:
        raise ValueError("Attention!\nVotre matrice ne doit pas être symétrique")

    for k in range(n-1):
        pivot_indice = pivot(matrice, k, min(k + m + 1, n))
        if pivot_indice != k:
         echanger_lignes(matrice, b, k, pivot_indice)

        if matrice[k][k] == 0:
         print("Il existe une certaine division par zéro")
         continue
        for i in range(k +1, min(k + m + 1, n)): 
         matrice[i][k]=matrice[i][k]/matrice[k][k]
         for j in range(k+1 , n):
            matrice[i][j] -= matrice[i][k]*matrice[k][j]
         b[i] -= matrice[i][k] * b[k]

    x = res_sup_densee(matrice, b, n)
    return x

def Gauss_dense(matrice,b,n):

    if(not symetrique(matrice)):
        raise ValueError("Attention!\nVotre matrice doit être symétrique")
    print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))

    for k in range(n):
        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice doit être une matrice définie positive")
    for k in range(n-1):
        for i in range(k+1,n):
            matrice[i][k]=matrice[i][k]/matrice[k][k]
            for j in range(i, n):  # Commencez par i pour maintenir la symétrie
                matrice[i][j] -= matrice[i][k]*matrice[k][j]
                if i != j:
                 matrice[j][i] = matrice[i][j]  # Reflet de la symétrie
            b[i] -= matrice[i][k] * b[k]

    x=res_sup_densee(matrice,b,n)
    return x

def Gauss_bande(matrice,b,n,largeur):
    m=(largeur-1)//2
    if not matrice_bande(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")

    if not symetrique(matrice):
        raise ValueError("Attention!\nVotre matrice n'est pas symétrique")
       
    '''print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))'''

    for k in range(n):
        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")
            
    for k in range(n-1):
     for i in range(k+1, min(k+m+1, n)):
        matrice[i][k] /= matrice[k][k]
        for j in range(k+1, min(k+m+1, n)):
            matrice[i][j] -= matrice[i][k] * matrice[k][j]
            if i != j:  # Pour éviter de réécrire la diagonale
                matrice[j][i] = matrice[i][j]
        b[i] -= matrice[i][k] * b[k]

    x=res_sup_densee(matrice,b,n)
    return x

def echanger_lignes(matrice, b, i, j):
    matrice[i], matrice[j] = matrice[j], matrice[i]
    b[i], b[j] = b[j], b[i]

def determinant(matrice):

    if len(matrice)==1:
        return matrice[0][0]

    if len(matrice)==2:
        return matrice[0][0]*matrice[1][1]-matrice[0][1]*matrice[1][0]

    det=0
    for i in range(len(matrice)):
        sous=decomposition_sous_matrice_determinant(matrice,i)
        det+=((-1)**i)*matrice[0][i]*determinant(sous)
    return det

def decomposition_sous_matrice_determinant(matrice, colonne_a_supprimer):
    return [ligne[:colonne_a_supprimer]+ligne[colonne_a_supprimer+1:] for ligne in matrice[1:]]

def decomposition_sous_matrice(matrice,k):

    return [ a[:k] for a in matrice[:k] ]

def Cholesky_dense(matrice,b,n):
    '''if not (symetrique(matrice)):
        raise ValueError("Attention!\nVotre matrice n'est pas symétrique")
    print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))
    for k in range(n):
        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")'''

    L = [[0.0] * n for _ in range(n)]

    for j in range(n):
        L[j][j] = matrice[j][j]
        for k in range(j):
            L[j][j] -= L[j][k] ** 2
        L[j][j] =math.sqrt(L[j][j])

        for i in range(j+1, n):
            L[i][j] = matrice[i][j]
            for k in range(j):
                L[i][j] -= L[i][k] * L[j][k]
            L[i][j] /= L[j][j]

    y=res_inf_densee(L,b,n)
    x=res_sup_densee(transopse(L,n),y,n)

    return x

def Cholesky_bande(matrice,b,n,largeur):

    m=(largeur-1)//2
    if not matrice_bande(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
    '''if not (symetrique(matrice)):
        raise ValueError("Attention!\nVotre matrice n'est pas symétrique")'''

    '''print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))'''

    '''for k in range(n):

        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")'''

    L = [[0.0] * n for _ in range(n)]

    for j in range(n):
        L[j][j] = matrice[j][j]
        for k in range(max(0, j - m),j): #on considère que la bande
            L[j][j] -= L[j][k] ** 2
        L[j][j] =math.sqrt(L[j][j])

        for i in range(j+1, min(n, j + m+1)):
            L[i][j] = matrice[i][j]
            for k in range(max(0, j - m),j):
                L[i][j] -= L[i][k] * L[j][k]
            L[i][j] /= L[j][j]

    y=res_inf_densee(L,b,n)
    x=res_sup_densee(transopse(L,n),y,n)
    return x

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.01):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def multiply_banded_matrices(A, B):  
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]   
    for i in range(rows_A):
        for j in range(cols_B):           
            for k in range(max(0, i - len(A) + 1), min(cols_A, i + 1)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def set_matrix_gauss_seidel(matrix,n):
    matrix_columns = len(matrix[0])  
    matrix_used = [[0.0 for _ in range(matrix_columns)] for _ in range(n)]
    F = [[0.0 for _ in range(matrix_columns)] for _ in range(n)]
    for i in range(n):
        for j in range(0, i + 1):
            matrix_used[i][j] = matrix[i][j]
        
        for j in range(i + 1, matrix_columns):
            F[i][j] = matrix[i][j]
    
    try:
        matrix_used_inverse = np.linalg.inv(matrix_used)
    
    except np.linalg.LinAlgError:
        raise ValueError("message: La matrice utilisée pour obtenir la matrice gauss seidel n'est pas inversible.")

    seidel_matrix = multiply_banded_matrices(matrix_used_inverse, F)

    return seidel_matrix

def solve_gauss_seidel_with_epsilon(matrix, vector, epsilon,n):
    maximum = 0
    matrix_rows = n
    y = [[0] for _ in range(matrix_rows)]
    seidel_matrix = set_matrix_gauss_seidel(matrix,n)
    eigenvalue, vectors = np.linalg.eig(seidel_matrix)
    if max(eigenvalue) >= 1:
        raise ValueError("message: La matrice est divergente.")

    while True:
        for i in range(matrix_rows):
            s = 0

            for j in range(matrix_rows):
                if j != i:
                    s += matrix[i][j] * y[j][0]
            
            s = (vector[i] - s) / matrix[i][i]
            if maximum < (abs_result := abs(y[i][0] - s)):
                maximum = abs_result
            
            y[i][0] = s
        
        if maximum > epsilon:
            break

    return y    

def solve_gauss_seidel_with_max_iteration(matrix, vector, max_iteration):
    maximum = 0
    matrix_rows = len(matrix)
    y = [[0] for _ in range(matrix_rows)]
    counter = 0
    seidel_matrix = set_matrix_gauss_seidel(matrix,len(matrix))
    eigenvalue, vectors = np.linalg.eig(seidel_matrix)
    if max(eigenvalue) >= 1:
        raise ValueError("message: La matrice est divergente.")

    for k in range(max_iteration):
        for i in range(matrix_rows):
            s = 0
            for j in range(matrix_rows):
                if j != i:
                    s += matrix[i][j] * y[j][0]            
            s = (vector[i] - s) / matrix[i][i]
            if maximum < (abs_result := abs(y[i][0] - s)):
                maximum = abs_result           
            y[i][0] = s

    return y

def set_matrix_jacobi(matrix):
    matrix_rows = len(matrix)
    matrix_columns = len(matrix[0])
    jacobi_matrix = [[0.0 for _ in range(matrix_columns)] for _ in range(matrix_rows)]
    for i in range(matrix_rows):
        for j in range(matrix_columns):
            if i != j:
                jacobi_matrix[i][j] = - (matrix[i][j] / matrix[i][i])

    return jacobi_matrix

def solve_jacobi_with_epsilon(matrix, vector, epsilon):
    matrix_rows = len(matrix)
    x = [[0] for _ in range(matrix_rows)]
    y = [[0] for _ in range(matrix_rows)]
    jacobi_matrix = set_matrix_jacobi(matrix)
    eigenvalue, vectors = np.linalg.eig(jacobi_matrix)
    if max(eigenvalue) >= 1:
        raise ValueError("message: La matrice est divergente.")
    while True:
        for i in range(matrix_rows):
            x[i][0] = y[i][0]

        for i in range(matrix_rows):
            s = vector[i]

            for j in range(matrix_rows):
                if i != j:
                    s -= matrix[i][j] * x[j][0]

            y[i][0] = s / matrix[i][i]

        if max(abs(x[0] - y[0]) for y, x in zip(y, x)) > epsilon:
            break
            
    return y

def solve_jacobi_with_max_iteration(matrix, vector, max_iteration):
    matrix_rows = len(matrix)
    x = [[0] for _ in range(matrix_rows)]
    y = [[0] for _ in range(matrix_rows)]
    counter = 0
    jacobi_matrix = set_matrix_jacobi(matrix)
    eigenvalue, vectors = np.linalg.eig(jacobi_matrix)
    if max(eigenvalue) >= 1:
        raise ValueError("message: La matrice est divergente.")

    for k in range(max_iteration):
        for i in range(matrix_rows):
            x[i][0] = y[i][0]
        for i in range(matrix_rows):
            s = vector[i]
            for j in range(matrix_rows):
                if i != j:
                    s -= matrix[i][j] * x[j][0]

            y[i][0] = s / matrix[i][i]
            
    return y

def gauss_jordan_elimination(M, V):
    if(symetrique(M)==False):
        raise ValueError("Attention!\nVotre matrice n'est pas symétrique")
    for k in range(len(M)):
        if determinant(decomposition_sous_matrice(M, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")
    augmented_matrix = [row + [V[i]] for i, row in enumerate(M)]
    rows, cols = len(augmented_matrix), len(augmented_matrix[0])

    for i in range(rows):
        # Recherche du pivot dans la colonne i
        pivot_row = i
        for k in range(i + 1, rows):
            if abs(augmented_matrix[k][i]) > abs(augmented_matrix[pivot_row][i]):
                pivot_row = k
        augmented_matrix[i], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[i]

        pivot = augmented_matrix[i][i]
        for j in range(i, cols):
            augmented_matrix[i][j] /= pivot
        
        for j in range(rows):
            if j != i:
                factor = augmented_matrix[j][i]
                for k in range(i, cols):
                    augmented_matrix[j][k] -= factor * augmented_matrix[i][k]
    result_vector = [row[-1] for row in augmented_matrix]

    return result_vector

def produit_matrice_bande_demi_inf(matrice_bandee,n1, matrice_demi_inf, largeur):
    m=(largeur-1)//2
    if not matrice_bande(matrice_bandee,n1,largeur):
        raise ValueError("Attention!\nVotre première matrice n'est pas une matrice bande")
        
    if not matrice_demi_bande_inf(matrice_demi_inf,n1,m+1):
        raise ValueError("Attention!\nVotre deuxième matrice n'est pas une matrice demi bande inférieure ou sa largeur est différent à m+1\n veuillez vérifier")

    resultat = [[0 for _ in range(len(matrice_demi_inf[0]))] for _ in range(len(matrice_bandee))]
    
    for i in range(n1):
        for j in range(n1):
            for k in range(max(0, i - m), min(n1, i + m + 1)):
                    resultat[i][j] += matrice_bandee[i][k] * matrice_demi_inf[k][j]

    return resultat

def produit_matrice_demi_bande_inf_sup(A, B, s, r):   
    s=s-1
    r=r-1   
    if (s==r):
        raise ValueError("S et R doivent être différents")
    
    n=len(A)
    p=len(B[0])
    if not matrice_demi_bande_sup(B,n,r+2):
        raise ValueError("Attention!\nVotre première matrice n'est pas une matrice demi-bande inférieure")
    if not matrice_demi_bande_inf(A,n,s+2):
        raise ValueError("Attention!\nVotre deuxième matrice n'est pas une matrice demi-bande supérieure")
    
    if (s<r):
        m=r
    else:
        m=s

    resultat = [[0 for _ in range(n)] for _ in range(p)]
    for i in range(n):
        for j in range(p):
            for k in range(max(0, i - m), min(n, i + m + 1)):
                resultat[i][j] += A[i][k] * B[k][j]
  
    return resultat

def multiply_row(matrix, row_index, scalar):
    matrix[row_index] = [element * scalar for element in matrix[row_index]]    

def gauss_jordan(a,n,largeur):   
   '''if not matrice_bande(a,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")'''
   
   m=len(a)#nombre de ligne
   n=len(a[0])#nombre de colonne  
   aug = [row + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(a)]
   n=len(aug[0])
   for z in range(m): 
         r=z
         for j in range(z,m):
             if(abs(aug[z][z])<abs(aug[j][z])):
                 r=j
         if(r!=z):
             aug[z],aug[r]=aug[r],aug[z]
         if(aug[z][z]!=0):
           p=1/aug[z][z]
         else:
             p=1  
         multiply_row(aug,z, p)
         for b in range(m):
             if(b!=z):
                 p=aug[b][z]
                 for a in range(n):
                     aug[b][a]=aug[b][a]-p*aug[z][a] 
                     
   inverse_matrix = [row[(-n//2):] for row in aug]
   return inverse_matrix
   

def multiply_matrices_matrice_inverse_gauus_jordan(matrix1,n1,largeur):
    if not matrice_bande(matrix1,n1,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
    m=(largeur-1)/2
    m1=gauss_jordan(matrix1,n1,largeur)
    if len(matrix1[0]) != len(m1):
        print("Le nombre de colonnes de la première matrice doit être égal au nombre de lignes de la deuxième matrice.")
        return None
    result = [[0 for _ in range(len(m1[0]))] for _ in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(m1[0])):
            for k in range(len(m1)):
                temp_result = matrix1[i][k] * m1[k][j]
            # Arrondir pour éviter -0
                temp_result = round(temp_result, 10)
                result[i][j] += temp_result
            # Correction pour -0 à 0
                if result[i][j] == 0:                  
                  result[i][j] = abs(result[i][j])

    return {'solution': result, 'm1': m1 }

def transpose_matrix(matrix,n,largeur):
    if not matrice_bande(matrix,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
    cols = len(matrix[0])
    transposed_matrix = [[0 for _ in range(n)] for _ in range(cols)]
    for i in range(n):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

def matrix_fois_matrice_transpose(m,n,largeur):
    m1=transpose_matrix(m,n,largeur)
    x = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x[i][j] += m[i][k] * m1[k][j]
    return {'solution': x, 'm1': m1 }        

def produit_matrice_vecteur(matrice, vecteur,n):
    resultat = [0] * n
    for i in range(n):
        for j in range(n):
            resultat[i] += matrice[i][j] * vecteur[j]

    return resultat

def produit_matrice_triangulaire_inferieure_vecteur(matrice, vecteur,n):

    if not est_triangulaire_inferieure(matrice):
                raise ValueError("Attention!\nVotre matrice n'est pas une matrice triangulaire inférieure")
   
    resultat = [0] * n
    for i in range(n):
        for j in range(i+1):
            resultat[i] += matrice[i][j] * vecteur[j] 
    return resultat

def produit_matrice_triangulaire_superieure_vecteur(matrice, vecteur, n):
    if not est_triangulaire_superieure(matrice):
                raise ValueError("Attention!\nVotre matrice n'est pas une matrice triangulaire supérieure")
    resultat = [0] * n
    for i in range(n-1, -1, -1):
        for j in range(i,n):
            resultat[i] += matrice[i][j] * vecteur[j]
    return resultat

def produit_matrice_demi_bande_superieure_vecteur(matrice, vecteur, n, largeur):
    m=largeur-1
    resultat = [0] * n
    if not matrice_demi_bande_sup(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice demi-bande supérieure")

    for i in range(n):
        for j in range(i, min(i+1+m, n)):
            resultat[i] += matrice[i][j] * vecteur[j]

    return resultat

def produit_matrice_triangulaire_inferieure_demi_bande_vecteur(matrice, vecteur, n, largeur):
    demi_bande=largeur-1
    
    resultat = [0] * n
    if not matrice_demi_bande_inf(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice demi-bande inférieure")
 
    for i in range(n):
        for j in range(max(0, i - demi_bande ), i + 1):
            resultat[i] += matrice[i][j] * vecteur[j]

    return resultat


@app.route('/', methods=['POST', 'GET'])

def index():
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        method = request.form.get('method')
        
        n = int(request.form.get('matrix-size', 3))
        largeur=safe_int(request.form.get('bandwidth-general', 0))
        s=safe_int(request.form.get('bandwidth-s'),0)
        r=safe_int(request.form.get('bandwidth-r'), 0)
        
        epsilon = request.form.get('bandwidth-e', type=float)
        #print(s)
        #print(r)
        max_iteration=safe_int(request.form.get('bandwidth-m', 0))
        matrix = [[safe_int(request.form.get(f'A-{i}-{j}', 0)) for j in range(n)] for i in range(n)]
        vector = [safe_int(request.form.get(f'B-{i}', 0)) for i in range(n)]
        matrix1_rows = safe_int(request.form.get('matrix2-rows', 0))
        matrix1_cols = safe_int(request.form.get('matrix2-cols', 0))
        matrix1 = [[safe_int(request.form.get(f'B-{i}-{j}', 0)) for j in range(matrix1_cols)] for i in range(matrix1_rows)]
        rows = int(request.form.get('matrix-rows', 3))
        cols = int(request.form.get('matrix-cols', 3))
        matrix3 = [[safe_int(request.form.get(f'A-{i}-{j}', 0)) for j in range(cols)] for i in range(rows)]
        #print(request.form.get('bandwidth-s'))

        try:
            result = None
            if method == "res_sup_dense":
                result = res_sup_dense(matrix, vector, n)
                print(matrix)
                print(vector)
                print(n)
            if method=="res_inf_dense":
                result=res_inf_dense(matrix,vector,n)
                print(matrix)
                print(vector)
                print(n)
            if method=="res_demi_bande_sup":                                     
                result=res_demi_bande_sup(matrix,vector,n,largeur)
                print(matrix)
                print(vector)
                print(n)
            if method=="res_demi_bande_inf":
                result=res_demi_bande_inf(matrix,vector,n,largeur)
                print(matrix)
                print(vector)
                print(n)
            if method=="LU_dense":
                result=LU_dense(matrix,vector,n)
                print(matrix)
                print(vector)
                print(n)
                formatted_result = {
                    'solution': result['solution'],
                    'L': [list(row) for row in result['L']],
                    'U': [list(row) for row in result['U']]
                }
                return jsonify(formatted_result)
                
            if method=="Gauss_pivotage_dense":
                result=Gauss_pivotage_dense(matrix,vector,n)
                print(matrix)
                print(vector)
                print(n)
            if method=="Gauss_dense":
                result=Gauss_dense(matrix,vector,n)
                print(matrix)
                print(vector)
                print(n)
            if method=="Cholesky_dense":
                result=Cholesky_dense(matrix,vector,n)
                print(matrix)
                print(vector)
                print(n)
            if method=="Cholesky_bande":
                result=Cholesky_bande(matrix,vector,n,largeur)
                print(matrix)
                print(vector)
                print(n)
            if method=="Gauss_bande":
                result=Gauss_bande(matrix,vector,n,largeur)
                print(matrix)
                print(vector)
                print(n)
            if method=="Gauss_pivotage_bande":
                result=Gauss_pivotage_bande(matrix,vector,n,largeur)
                print(matrix)
                print(vector)
                print(n)
            if method=="LU_bande":
                result=LU_bande(matrix,n,vector,largeur)
                print(matrix)
                print(vector)
                print(n)
                formatted_result = {
                    'solution': result['solution'],
                    'L': [list(row) for row in result['L']],
                    'U': [list(row) for row in result['U']]
                }
                return jsonify(formatted_result)
            if method=="gauss_seidel_with_epsilon":
                result=solve_gauss_seidel_with_epsilon(matrix, vector, epsilon,n)
                print(matrix)
                print(vector)
                print(epsilon)
                print(n)
            if method=="gauss_seidal_with_max_iteration":
                result=solve_gauss_seidel_with_max_iteration(matrix, vector, max_iteration)
                print(matrix)
                print(vector)
                print(max_iteration)
            if method=="jacobi_with_epsilon":
                result=solve_jacobi_with_epsilon(matrix, vector, epsilon)
                print(matrix)
                print(vector)
                print(epsilon)
            if method=="jacobi_with_max_iteration":
                result=solve_jacobi_with_max_iteration(matrix, vector, max_iteration)
                print(matrix)
                print(vector)
                print(max_iteration)
            if method=="gauss_jordan":
                result=gauss_jordan_elimination(matrix,vector)
                print(matrix)
                print(vector)
            if method=="multiplication_matrice_demi_bande_inferieur":
                print(matrix)
                print(matrix1)
                print(n)
                print(largeur)
                result=produit_matrice_bande_demi_inf(matrix,n, matrix1, largeur)
                
            if method=="produit_matrice_demi_bande_inf_largeur_different ":
                result=produit_matrice_demi_bande_inf_sup(matrix, matrix1, s, r)
                print(result)
            if method=="matrix_fois_matrice_transpose":
                result= matrix_fois_matrice_transpose(matrix3,len(matrix3),largeur)
                print(matrix3)
                print(matrix1)
                print(n)
                print(result)
                formatted_result = {
                    'solution': result['solution'],
                    'm1': [list(row) for row in result['m1']]
                }
                return jsonify(formatted_result)
            if method=="produit_de_matrice_fois_inverse":
                print(matrix3)
                print(matrix1)
                print(n)
                result=multiply_matrices_matrice_inverse_gauus_jordan(matrix3,n,largeur)
                formatted_result = {
                    'solution': result['solution'],
                    'm1': [list(row) for row in result['m1']]
                }
                return jsonify(formatted_result)
            if method=="produit_matrice_vecteur":
                result=produit_matrice_vecteur(matrix, vector,n)
                print(matrix)
                print(vector)
                print(n)
            if method=="produit_matrice_triangulaire_inferieure_vecteur":
                result=produit_matrice_triangulaire_inferieure_vecteur(matrix, vector,n)
                print(matrix)
                print(matrix1)
                print(matrix3)
                print(vector)
                print(n)
            if method=="produit_matrice_triangulaire_superieure_vecteur":
                result=produit_matrice_triangulaire_superieure_vecteur(matrix, vector, n)
            if method=="produit_matrice_demi_bande_superieure_vecteur":
                result=produit_matrice_demi_bande_superieure_vecteur(matrix, vector, n,largeur)
            if method=="produit_matrice_triangulaire_inferieure_demi_bande_vecteur":
                result=produit_matrice_triangulaire_inferieure_demi_bande_vecteur(matrix, vector, n, largeur)  

            result = format_result(result)
            return jsonify(result=result, error_message=None)

        except ValueError as e:
            print("An error occurred:", e)
            return jsonify(result=None, error_message=str(e))

    return render_template('index.html', n=3)

    # Format the result based on its structure
    return str(result)
    if request.method == 'POST':
        method = request.form.get('method')
        n = int(request.form.get('matrix-size', 3))
        
        matrix = [[safe_int(request.form.get(f'A-{i}-{j}', 0)) for j in range(n)] for i in range(n)]
        vector = [safe_int(request.form.get(f'B-{i}', 0)) for i in range(n)]

        try:
            result = None
            if method == "res_sup_dense":
                result = res_sup_dense(matrix, vector, n)
            # Add other method conditions here

            # Convert the result to a suitable format for JSON response
            result = format_result(result)  # Define this function to format your result

            return jsonify(result=result, error_message=None)

        except ValueError as e:
            return jsonify(result=None, error_message=str(e))

    # If it's a GET request, just render the template
    return render_template('index.html', n=3)

def format_result(result):
    return result


if __name__=="__main__":
    app.run(debug==True)




