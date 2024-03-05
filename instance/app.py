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

    #if not est_triangulaire_superieure(matrice):
                #raise ValueError("Attention!\nVotre matrice n'est pas une matrice triangulaire supérieure")


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

def res_demi_bande_sup(matrice, b, n, m):
    
    x = [0] * n
    for i in range(n - 1, -1, -1):
        if matrice[i][i] == 0:
            raise ValueError("Le diagonal ne doit pas etre nul")

        x[i] = b[i]
        for j in range(i + 1, min(i + m + 1, n)):
            x[i] -= matrice[i][j] * x[j]
        x[i] /= matrice[i][i]
    return x

def res_demi_bande_inf(matrice, b, n, m):
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

#largeur=2m+1
#m est le nb de diagonaux au dessus ou au dessous du diagonal (et non pas la somme des deux)
def matrice_bande(matrice,n,largeur):
    m=(largeur-1)//2 #division d'entier
    if (m>(n-1)/2):
        raise ValueError("Attention!\nLa largeur de bande spécifiée est trop grande pour cette matrice")
        
    for i in range(n):
        for j in range(n):
            if abs(i-j)>m and matrice[i][j]!=0:
                raise ValueError("Attention!\nVotre matrice n'est une matrice bande")
                
    return True

def LU_dense(matrice,b,n):
    '''print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))'''

    for k in range(n):

        if determinant(decomposition_sous_matrice(matrice, k+1)) == 0:
            raise ValueError("Attention!\nVotre matrice n'admet pas une décomposition LU")
            
        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")
            
    print("\n")

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
    y=res_inf_dense(L,b,n) #Méthode descente
    x=res_sup_dense(U,y,n) #Méthode remontée
    print("La solution X=")

    return x

def LU_bande(matrice,n,b,largeur):

    if not matrice_bande(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
    

    '''print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))'''

    for k in range(n):

        if determinant(decomposition_sous_matrice(matrice, k+1)) == 0:
            raise ValueError("Attention!\nVotre matrice n'admet pas une décomposition LU")
            
        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")

           
    #print("\n")

    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1

        for j in range(i,min(i+largeur, n)):  # Exploitez la structure de bande ici
            somme_u = 0
            for k in range(max(i-largeur+1, 0),i):  # Limitez k à la largeur de la bande
                somme_u += U[k][j] * L[i][k]
            U[i][j] = matrice[i][j] - somme_u

        for j in range(i+1,min(i+largeur,n)):  # Encore, utilisez la largeur de la bande
            #if U[i][i] != 0:
            L[j][i] = U[i][j] / U[i][i]

    y=res_inf_dense(L,b,n) #Méthode descente
    x=res_sup_dense(U,y,n) #Méthode remontée
    #print("La solution X=")

    return x

def Gauss_pivotage_dense(matrice,b,n):

    if(symetrique(matrice)==True):
        raise ValueError("Attention!\nVotre matrice est symétrique")
        

    for k in range(n-1):
        pivot_indice = pivot(matrice, k, n)
        if pivot_indice != k:
            echanger_lignes(matrice, b, k, pivot_indice)
        for i in range(k+1,n):
            matrice[i][k]=matrice[i][k]/matrice[k][k]
            for j in range(k+1,n):
                matrice[i][j] -= matrice[i][k]*matrice[k][j]
            b[i] -= matrice[i][k] * b[k]

    x=res_sup_dense(matrice,b,n)
    return x

def Gauss_pivotage_bande(matrice,b,n,largeur):
    m=(largeur-1)//2
    if not matrice_bande(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
        


    if symetrique(matrice)==True:
        raise ValueError("Attention!\nVotre matrice est pas symétrique")
        


    for k in range(n-1):


        pivot_indice = pivot(matrice, k, min(k + m + 1, n))
        if pivot_indice != k:
         echanger_lignes(matrice, b, k, pivot_indice)

        if matrice[k][k] == 0:
         print("Il existe une certaine division par zéro")
         continue
        for i in range(k +1, min(k + m + 1, n)): #Cela garantit que vous ne traitez que les lignes qui se trouvent dans la bande spécifiée par m
         matrice[i][k]=matrice[i][k]/matrice[k][k]
         for j in range(k+1 , n):
            matrice[i][j] -= matrice[i][k]*matrice[k][j]
         b[i] -= matrice[i][k] * b[k]

    x = res_sup_dense(matrice, b, n)
    return x

def Gauss_dense(matrice,b,n):

    if(not symetrique(matrice)):
        raise ValueError("Attention!\nVotre matrice n'est pas symétrique")
        

    print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))

    for k in range(n):

        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")
            

    for k in range(n-1):
        for i in range(k+1,n):
            matrice[i][k]=matrice[i][k]/matrice[k][k]
            for j in range(i, n):  # Commencez par i pour maintenir la symétrie
                matrice[i][j] -= matrice[i][k]*matrice[k][j]
                if i != j:
                 matrice[j][i] = matrice[i][j]  # Reflet de la symétrie
            b[i] -= matrice[i][k] * b[k]

    x=res_sup_dense(matrice,b,n)
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
     for i in range(k+1, min(k+m+1, n)):  # garantit que vous ne traitez que les éléments qui se trouvent dans la bande spécifiée par m
        matrice[i][k] /= matrice[k][k]
        for j in range(k+1, min(k+m+1, n)):
            matrice[i][j] -= matrice[i][k] * matrice[k][j]
            if i != j:  # Pour éviter de réécrire la diagonale
                matrice[j][i] = matrice[i][j]  # Exploiter la symétrie
        b[i] -= matrice[i][k] * b[k]

    x=res_sup_dense(matrice,b,n)
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

def decomposition_sous_matrice(matrice,k):#k est le nombre des lignes et des colonnes à supprimer

    return [ a[:k] for a in matrice[:k] ] # matrice[:k]: les k premiers lignes et a[:k] prend les k premiers éléments

def Cholesky_dense(matrice,b,n):

    if not (symetrique(matrice)):
        raise ValueError("Attention!\nVotre matrice n'est pas symétrique")
        

    '''print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))'''

    for k in range(n):

        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")
            return False
    print("\n")

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

    y=res_inf_dense(L,b,n)
    x=res_sup_dense(transopse(L,n),y,n)

    return x

def Cholesky_bande(matrice,b,n,largeur):

    m=(largeur-1)//2
    #print(m)
    if not matrice_bande(matrice,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
        

    if not (symetrique(matrice)):
        raise ValueError("Attention!\nVotre matrice n'est pas symétrique")
        return False

    '''print("Les sous matrices sont:")
    for k in range(n):
        print (decomposition_sous_matrice(matrice, k+1))'''

    for k in range(n):

        if determinant(decomposition_sous_matrice(matrice, k+1)) <= 0:
            raise ValueError("Attention!\nVotre matrice n'est pas une matrice définie positive")
            return False
    print("\n")

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

    y=res_inf_dense(L,b,n)
    x=res_sup_dense(transopse(L,n),y,n)
    print("La solution X=")

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
    # Getting matrix rows and matrix columns
    matrix_columns = len(matrix[0])
    
    # Inialization of jacobi matrix
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
        raise ValueError("message: La matrice utilise pour obtenir la matrice gauss seidel n'est pas inversible.")

    seidel_matrix = multiply_banded_matrices(matrix_used_inverse, F)

    return seidel_matrix
#lezem narawha!!!!!
def solve_gauss_seidel_with_epsilon(matrix, vector, epsilon,n):
    # Initialization of max, result and counter
    maximum = 0
    matrix_rows = n
    y = [[0] for _ in range(matrix_rows)]

    # Testing the convergence of the matrix
    seidel_matrix = set_matrix_gauss_seidel(matrix,n)
    eigenvalue, vectors = np.linalg.eig(seidel_matrix)
    if max(eigenvalue) >= 1:
        raise ValueError("message: La matrice est divergente.")

    # Solving matrix
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
#lezem narawha!!!!!
def solve_gauss_seidel_with_max_iteration(matrix, vector, max_iteration):
   
    # Initialization of max, result and counter
    maximum = 0
    matrix_rows = len(matrix)
    y = [[0] for _ in range(matrix_rows)]
    counter = 0

    # Testing the convergence of the matrix
    seidel_matrix = set_matrix_gauss_seidel(matrix,len(matrix))
    eigenvalue, vectors = np.linalg.eig(seidel_matrix)
    if max(eigenvalue) >= 1:
        raise ValueError("message: La matrice est divergente.")

    # Solving matrix
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
    # Getting matrix rows and matrix columns
    matrix_rows = len(matrix)
    matrix_columns = len(matrix[0])

    # Inialization of jacobi matrix
    jacobi_matrix = [[0.0 for _ in range(matrix_columns)] for _ in range(matrix_rows)]

    for i in range(matrix_rows):
        for j in range(matrix_columns):
            if i != j:
                jacobi_matrix[i][j] = - (matrix[i][j] / matrix[i][i])

    return jacobi_matrix
#lezem narawha!!!!!
def solve_jacobi_with_epsilon(matrix, vector, epsilon):
    # Initialization of result and counter
    matrix_rows = len(matrix)
    x = [[0] for _ in range(matrix_rows)]
    y = [[0] for _ in range(matrix_rows)]

    # Testing the convergence of the matrix
    jacobi_matrix = set_matrix_jacobi(matrix)
    eigenvalue, vectors = np.linalg.eig(jacobi_matrix)
    if max(eigenvalue) >= 1:
        raise ValueError("message: La matrice est divergente.")
       
    # solving matrix
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
#lezem narawha!!!!!
def solve_jacobi_with_max_iteration(matrix, vector, max_iteration):
    # Initialization of result and counter
    matrix_rows = len(matrix)
    x = [[0] for _ in range(matrix_rows)]
    y = [[0] for _ in range(matrix_rows)]
    counter = 0

    # Testing the convergence of the matrix
    jacobi_matrix = set_matrix_jacobi(matrix)
    eigenvalue, vectors = np.linalg.eig(jacobi_matrix)
    if max(eigenvalue) >= 1:
        raise ValueError("message: La matrice est divergente.")
       
    # solving matrix
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
    # Combinez la matrice M avec le vecteur V pour former une matrice augmentée
    augmented_matrix = [row + [V[i]] for i, row in enumerate(M)]

    rows, cols = len(augmented_matrix), len(augmented_matrix[0])

    for i in range(rows):
        # Recherche du pivot dans la colonne i
        pivot_row = i
        for k in range(i + 1, rows):
            if abs(augmented_matrix[k][i]) > abs(augmented_matrix[pivot_row][i]):
                pivot_row = k

        # Échange des lignes pour avoir le pivot non nul
        augmented_matrix[i], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[i]

        # Normalisation de la ligne du pivot
        pivot = augmented_matrix[i][i]
        for j in range(i, cols):
            augmented_matrix[i][j] /= pivot
        
        # Élimination des autres lignes
        for j in range(rows):
            if j != i:
                factor = augmented_matrix[j][i]
                for k in range(i, cols):
                    augmented_matrix[j][k] -= factor * augmented_matrix[i][k]
    # Récupération du vecteur résultant de la matrice augmentée
    result_vector = [row[-1] for row in augmented_matrix]

    return result_vector

def produit_matrice_bande_demi_inf(matrice_bandee,n1, matrice_demi_inf, largeur):
    # VÃ©rifier que les dimensions sont compatibles
    m=(largeur-1)//2
    '''if not matrice_bande(matrice_bandee,n1,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")'''
        
    '''if len(matrice_bandee[0]) != len(matrice_demi_inf):
        raise ValueError("Les dimensions des matrices ne sont pas compatibles.")'''

    # Initialiser la matrice rÃ©sultante avec des zÃ©ros
    resultat = [[0 for _ in range(len(matrice_demi_inf[0]))] for _ in range(len(matrice_bandee))]
    

    # Parcourir chaque Ã©lÃ©ment de la matrice rÃ©sultante
    for i in range(n1):
        for j in range(n1):
            # Parcourir les Ã©lÃ©ments non nuls de la matrice bande
            #max loula 5ater aka les 0 fil matrice bande ili milouta
            #min amelneha al les 0 fil fil matrice bande mil fou9
            #ili howa nafs lahkeya lil matrice bande inf ala5ater madhroubin donc les zeros dans matrice bande sont unitile de le multiplie
            for k in range(max(0, i - m), min(n1, i + m + 1)):
                    resultat[i][j] += matrice_bandee[i][k] * matrice_demi_inf[k][j]

    return resultat


def produit_matrice_demi_bande_inf_sup(A, B, s, r):
    s=s-1
    r=r-1
    
    if (s==r):
        raise ValueError("S et R doivent etre différents")

    # VÃ©rifier que les dimensions sont compatibles
    
    n=len(A)
    p=len(B[0])
    #if len(n) != len(p):
        #raise ValueError("Les dimensions des matrices ne sont pas compatibles.")
    if (s<r):
        m=r
    else:
        m=s
    # Initialiser la matrice rÃ©sultante avec des zÃ©ros
    resultat = [[0 for _ in range(n)] for _ in range(p)]
    for i in range(n):
        for j in range(p):
            for k in range(max(0, i - m), min(n, i + m + 1)):
                resultat[i][j] += A[i][k] * B[k][j]

    # Parcourir chaque Ã©lÃ©ment de la matrice rÃ©sultante
    

    return resultat

def multiply_row(matrix, row_index, scalar):
    matrix[row_index] = [element * scalar for element in matrix[row_index]]    

def gauss_jordan(a,n,largeur):
   
   if not matrice_bande(a,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
   
   m=len(a)#nombre de ligne
   n=len(a[0])#nombre de colonne
   if(n==m):    
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
                 for a in range(n):#nbr de colonnes
                     aug[b][a]=aug[b][a]-p*aug[z][a] 
                     
     inverse_matrix = [row[(-n//2):] for row in aug]

   else:
       print("erreur la matrice n'est pas carre")
   return inverse_matrix

def multiply_matrices_matrice_inverse_gauus_jordan(matrix1,n1,largeur):
    # Vérifiez si les matrices peuvent être multipliées
   
    m=(largeur-1)/2
    matri=gauss_jordan(matrix1,n1,largeur)
    if len(matrix1[0]) != len(matri):
        print("Le nombre de colonnes de la première matrice doit être égal au nombre de lignes de la deuxième matrice.")
        return None

    # Initialiser une matrice résultante remplie de zéros
    result = [[0 for _ in range(len(matri[0]))] for _ in range(len(matrix1))]

    # Effectuer la multiplication
    for i in range(len(matrix1)):
        for j in range(len(matri[0])):
            for k in range(len(matri)):
                result[i][j] += matrix1[i][k] * matri[k][j]

    return result

def transpose_matrix(matrix,n,largeur):
    if not matrice_bande(matrix,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
    # Trouver les dimensions de la matrice
    cols = len(matrix[0])

    # Créer une matrice vide avec les dimensions échangées
    transposed_matrix = [[0 for _ in range(n)] for _ in range(cols)]

    # Remplir la matrice transposée
    for i in range(n):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

def matrix_fois_matrice_transpose(m,n,largeur):
    if not matrice_bande(m,n,largeur):
        raise ValueError("Attention!\nVotre matrice n'est pas une matrice bande")
    m1=transpose_matrix(m,n,largeur)
    result = [[0 for _ in range(n)] for _ in range(len(m1))]
    for i in range(n):
        for j in range(len(m1)):
            for k in range(len(m1)):
                result[i][j] += m[i][k] * m1[k][j]
    return result        

def produit_matrice_vecteur(matrice, vecteur,n):
    resultat = [0] * n

    # Effectuer le produit matrice-vecteur
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
    # Vérifier que la matrice est carrée
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


    for i in range(n):
        for j in range(i, min(i+1+m, n)):
            resultat[i] += matrice[i][j] * vecteur[j]

    return resultat

def produit_matrice_triangulaire_inferieure_demi_bande_vecteur(matrice, vecteur, n, largeur):
    demi_bande=largeur-1
    
    resultat = [0] * n

 
    for i in range(n):
        for j in range(max(0, i - demi_bande ), i + 1):
            resultat[i] += matrice[i][j] * vecteur[j]

    return resultat

# Define your matrix operation functions like res_sup_dense here

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
        print(matrix)
        print(matrix1)
        print(vector)
        #print(request.form.get('bandwidth-s'))

        try:
            result = None
            #if method in ["Cholesky_bande", "Gauss_bande", "Gauss_pivotage_bande", "LU_bande"] and largeur <= 0:
                #raise ValueError("Une largeur valide est requise pour cette méthode.")
            if method == "res_sup_dense":
                result = res_sup_dense(matrix, vector, n)
            if method=="res_inf_dense":
                result=res_inf_dense(matrix,vector,n)
            if method=="res_demi_bande_sup":                                     
                result=res_demi_bande_sup(matrix,vector,n)
            if method=="res_demi_bande_inf":
                result=res_demi_bande_inf(matrix,vector,n)
            if method=="LU_dense":
                result=LU_dense(matrix,vector,n)
            if method=="Gauss_pivotage_dense":
                result=Gauss_pivotage_dense(matrix,vector,n)
            if method=="Gauss_dense":
                result=Gauss_dense(matrix,vector,n)
            if method=="Cholesky_dense":
                result=Cholesky_dense(matrix,vector,n)
            if method=="Cholesky_bande":
                result=Cholesky_bande(matrix,vector,n,largeur)
            if method=="Gauss_bande":
                result=Gauss_bande(matrix,vector,n,largeur)
            if method=="Gauss_pivotage_bande":
                result=Gauss_pivotage_bande(matrix,vector,n,largeur)
            if method=="LU_bande":
                result=LU_bande(matrix,n,vector,largeur)
            if method=="gauss_seidel_with_epsilon":
                result=solve_gauss_seidel_with_epsilon(matrix, vector, epsilon,n)
            if method=="gauss_seidal_with_max_iteration":
                result=solve_gauss_seidel_with_max_iteration(matrix, vector, max_iteration)
            if method=="jacobi_with_epsilon":
                result=solve_jacobi_with_epsilon(matrix, vector, epsilon)
            if method=="jacobi_with_max_iteration":
                result=solve_jacobi_with_max_iteration(matrix, vector, max_iteration)
            if method=="gauss_jordan":
                result=gauss_jordan_elimination(matrix,vector)
            if method=="multiplication_matrice_demi_bande_inferieur":
                print(matrix)
                print(matrix1)
                print("multiplication_matrice_demi_bande_inferieur")
                result=produit_matrice_bande_demi_inf(matrix,n, matrix1, largeur)
                
            if method=="produit_matrice_demi_bande_inf_largeur_different ":
                print("\n\n yech")
                print(matrix)
                print(matrix1)
                print("produit_matrice_demi_bande_inf_largeur_different ")
                result=produit_matrice_demi_bande_inf_sup(matrix, matrix1, s, r)
                
            if method=="matrix_fois_matrice_transpose":
                print("yech")
                print("matrix_fois_matrice_transpose")

                result= matrix_fois_matrice_transpose(matrix,n,largeur)
            if method=="produit_de_matrice_fois_inverse":
                print("yech")
                result=multiply_matrices_matrice_inverse_gauus_jordan(matrix1,n,largeur)
            if method=="produit_matrice_vecteur":
                print(vector)
                print("produit_matrice_vecteur")
                result=produit_matrice_vecteur(matrix, vector,n)
            if method=="produit_matrice_triangulaire_inferieure_vecteur":
                result=produit_matrice_triangulaire_inferieure_vecteur(matrix, vector,n)
            if method=="produit_matrice_triangulaire_superieure_vecteur":
                result=produit_matrice_triangulaire_superieure_vecteur(matrix, vector, n)
            if method=="produit_matrice_demi_bande_superieure_vecteur":
                result=produit_matrice_demi_bande_superieure_vecteur(matrix, vector, n,largeur)
            if method=="produit_matrice_triangulaire_inferieure_demi_bande_vecteur":
                result=produit_matrice_triangulaire_inferieure_demi_bande_vecteur(matrix, vector, n, largeur)  

            result = format_result(result)  # Format the result
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
    # Format the result here based on its structure
    # Example: Convert a list or matrix to a string
    return result


if __name__=="__main__":
    app.run(debug==True)




