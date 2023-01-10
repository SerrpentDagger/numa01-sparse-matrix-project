#%% -*- coding: utf-8 -*-

import copy
from matplotlib.pyplot import *
import numpy as np
import random

def randmatrix(approximate_sparsity, num_rows, num_cols):
    """
    

    Parameters
    ----------
    approximate_sparsity : integer
        Generates the expected ratio of zeros in relation to other values.
    num_rows : integer
        The number of rows in the randomly generated matrix.
    num_cols : integer
        The number of columns in the randomly generated matrix.

    Returns
    -------
    matrix : np.array
        A matrix is a matrix...

    """
    matrix = np.zeros(shape=[num_rows, num_cols])
    for i in range(num_rows):
        for j in range(num_cols):
            num = random.random()
            if num > approximate_sparsity:
                matrix[i,j] = random.randint(1,1000000)
    return matrix

# These three functions are used heavily in the switch method,
# and are hopefully useful for the addition method as well

def decompress(array, dimension):
    """
    

    Parameters
    ----------
    array : np.array
        An array of numbers that are non-decreasing.
    dimension : integer
        The number of rows or the number of columns, depending on which array is used.

    Returns
    -------
    np.array
        A decompressed array.

    """
    return np.repeat(np.arange(dimension), np.diff(array))

def compress(array, dimension, number_of_nonzero):
    """
    

    Parameters
    ----------
    array : np.array
        An array of numbers that are non-decreasing.
    dimension : integer
        The number of rows or the number of columns, depending on which array is used.
    number_of_nonzero : integer
        The number of non-zero elements.

    Returns
    -------
    np.array
        A compressed array.

    """
    # might replace this (and the other functions) with another version if I find something that works quicker
    values, indices = np.unique(array, return_index=True)
    return np.repeat([*indices, number_of_nonzero], np.diff([-1, *values, dimension]))

def reorder(first_indices, second_indices, values):
    """
    

    Parameters
    ----------
    first_indices : np.array
        An array.
    second_indices : np.array
        Another array.
    values : np.array
        Yet another array.

    Returns
    -------
    np.array
        Reordered version of first_indices.
    np.array
        Reordered version of second_indices.
    np.array
        Reordered version of values.

    """
    # reorders all three lists by first_indices, with second_indices as a tie-breaker
    new_order = np.lexsort((second_indices, first_indices))
    return first_indices[new_order], second_indices[new_order], values[new_order]

# Helps organize the print log.
def prnt(text):
    """
    

    Parameters
    ----------
    text : string
        Outputs a title for test cases.

    Returns
    -------
    None.

    """
    print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~", text)


class SparseMatrix:
    
    def __init__(self, matrix, tol = 10e-8):
        """
        

        Parameters
        ----------
        matrix : np.array
            A matrix is a matrix...
        tol : integer
            The value below which the data will be considered 0. The default is 10e-8.

        Returns
        -------
        None.

        """
        self.intern_represent = 'CSR'
        self.tol = tol
        
        matrix[np.abs(matrix) < tol] = 0
        
        self.v, self.cols, self.rows = [], [], [0]
        
        for r in matrix:
            nonzero_cols = np.nonzero(r)[0]
            self.cols.extend(nonzero_cols)
            self.v.extend(r[nonzero_cols])
            self.rows.append(len(self.cols))

        # This is a quick and probably bad fix for something I forgot to mention,
        # we assumed in the meeting that these three should be of type np.array
        self.v, self.cols, self.rows = np.array(self.v), np.array(self.cols), np.array(self.rows)    
       
        self.number_of_nonzero = len(self.cols)
        
        self.num_rows = matrix.shape[0]
        self.num_cols = matrix.shape[1]
        
    @staticmethod
    def toeplitz(n):
        """
        

        Parameters
        ----------
        n : integer
            The number of rows.

        Raises
        ------
        ValueError
            This error is raised when n is not positive.

        Returns
        -------
        toe : np.array
            A toeplitz matrix, i.e. a diagonal-constant matrix.

        """
        if n < 0:
            raise ValueError("The order of the Toeplitz matrix must be at least 0.")
        
        if not isinstance(n, int):
            raise TypeError("The order must be an integer.")

        toe = SparseMatrix(np.array([[]]))
        toe.num_rows = n
        toe.num_cols = n
        toe.tol = 10e-8
        
        toe.number_of_nonzero = 3*n - 2 if n > 0 else 0
        toe.v = np.array(([2, -1, -1] * n)[:-2])
        
        ind = np.array([0, 1, 0])
        
        toe.cols = []
        for i in range(n):
            toe.cols.extend(ind)
            ind += 1
        toe.cols = toe.cols[:-2]
        toe.cols = np.array(toe.cols)
        toe.rows = []
        if n == 0:
            return toe
        if n == 1:
            toe.rows = [0, 1]
            return toe
        
        toe.rows = [0, 2]
        toe.rows += [i*3 - 1 for i in range(2, n)]
        toe.rows += [toe.number_of_nonzero]
        toe.rows = np.array(toe.rows)
        
        return toe
    
    def switch(self, new_represent):
        """
        

        Parameters
        ----------
        new_represent : string
            The representation to convert to. CSR -> CSC.

        Returns
        -------
        None.

        """
        if new_represent != 'CSR' and new_represent != 'CSC':
            raise ValueError("Representation not supported.")

        if self.intern_represent == 'CSR' and new_represent == 'CSC':
            self.rows = decompress(self.rows, self.num_rows)
            self.cols, self.rows, self.v = reorder(self.cols, self.rows, self.v)
            self.cols = compress(self.cols, self.num_cols, self.number_of_nonzero)

        elif self.intern_represent == 'CSC' and new_represent == 'CSR':
            self.cols = decompress(self.cols, self.num_cols)
            self.rows, self.cols, self.v = reorder(self.rows, self.cols, self.v)
            self.rows = compress(self.rows, self.num_rows, self.number_of_nonzero)

        self.intern_represent = new_represent
        
    def change(self, i, j, value):
        """
        

        Parameters
        ----------
        i : index
            Input index representing rows.
        j : index
            Input index representing columns.
        value : index
            Input index representing a value.

        Raises
        ------
        IndexError
            This error is raised when the position is out of bounds.

        Returns
        -------
        None.

        """
        if i >= self.num_rows or j >= self.num_cols:
            raise IndexError("Position out of bounds.")
        if i < 0 or j < 0:
            raise IndexError("The coordinates of the position cannot be negative.")
        if not isinstance(i, int) or not isinstance(j, int):
            raise IndexError("Both indices must be integers.")

        nonZero = abs(value) >= self.tol
        if self.intern_represent == 'CSR':
            index = np.searchsorted(self.cols[self.rows[i]:self.rows[i+1]], j) + self.rows[i]
            # The reason for the try-except is that self.cols[index]
            # doesn't exist if index == self.number_of_nonzero
            if j < len(self.cols) and self.cols[index] == j and index != self.rows[i+1]:
                if nonZero:
                    self.v[index] = value
                else:
                    self.cols = np.delete(self.cols, index)
                    self.v = np.delete(self.v, index)
                    self.rows[i+1:] -= 1
                    self.number_of_nonzero -= 1
            elif nonZero:
                self.cols = np.insert(self.cols, index, j)
                self.v = np.insert(self.v, index, value)
                self.rows[i+1:] += 1
                self.number_of_nonzero += 1
        elif self.intern_represent == 'CSC':
            index = np.searchsorted(self.rows[self.cols[j]:self.cols[j+1]], i) + self.cols[j]
            if i < len(self.rows) and self.rows[index] == i and index != self.cols[j+1]:
                if nonZero:
                    self.v[index] = value
                else:
                    self.rows = np.delete(self.rows, index)
                    self.v = np.delete(self.v, index)
                    self.cols[j+1:] -= 1
                    self.number_of_nonzero -= 1
            elif nonZero:
                self.rows = np.insert(self.rows, index, i)
                self.v = np.insert(self.v, index, value)
                self.cols[j+1:] += 1
                self.number_of_nonzero += 1
                
                
    def equals(self, other):
        """
        

        Parameters
        ----------
        other : SparseMatrix
            A sparse matrix different from the self one.

        Raises
        ------
        ValueError
            This error is raised when something other than a sparse matrix is involved.

        Returns
        -------
        int
            It is an integer.

        """
		
        if isinstance(other, SparseMatrix):
            new = copy.copy(self)
            if self.intern_represent != other.intern_represent:
                new.switch(other.intern_represent)

            if np.sum(new != other) == 0:
                return 1
            else:
                return 0
        else:
            raise ValueError("Summand not of type SparseMatrix!")


    def add(self, other):
        """
        

        Parameters
        ----------
        other : SparseMatrix
            A sparse matrix different from the self one.

        Raises
        ------
        ValueError
            This error is raised when other is not a sparse matrix, 
            or when the dimensions are invalid.

        Returns
        -------
        Sum : SparseMatrix
            The output is a matrix that is the result of matrix addition.

        """
        
        if not isinstance(other, SparseMatrix):
            raise ValueError("other must be of type SparseMatrix!")
            
        if not (self.num_cols == other.num_cols and self.num_rows == other.num_rows):
                raise ValueError("Invalid dimensions!")

        Sum = copy.copy(self)
        Term = copy.copy(other)

        if Sum.intern_represent != Term.intern_represent:
            Sum.switch(Term.intern_represent)

        if Sum.intern_represent == 'CSR':

            Term.rows = decompress(Term.rows, Term.num_rows)
            Term.cols, Term.rows, Term.v = reorder(Term.cols, Term.rows, Term.v)
                
            Sum.rows = decompress(Sum.rows, Sum.num_rows)
            Sum.cols, Sum.rows, Sum.v = reorder(Sum.cols, Sum.rows, Sum.v)
                
        else:

            Sum.cols = decompress(Sum.cols, Sum.num_cols)
            Sum.rows, Sum.cols, Sum.v = reorder(Sum.rows, Sum.cols, Sum.v)

            Term.cols = decompress(Term.cols, Term.num_cols)
            Term.rows, Term.cols, Term.v = reorder(Term.rows, Term.cols, Term.v)
     
        Sum.rows = np.concatenate((Sum.rows, Term.rows))
        Sum.cols = np.concatenate((Sum.cols, Term.cols))
        Sum.v = np.concatenate((Sum.v, Term.v))
        Sum.rows, Sum.cols, Sum.v = reorder(Sum.rows, Sum.cols, Sum.v)
    
        coords = np.column_stack((Sum.cols,Sum.rows))
        i = -1
		
        while i < len(Sum.rows) -2:
            i+=1
            if (coords[i] == coords[i+1]).all():
                coords = np.delete(coords, i, axis=0)
                result = Sum.v[i] + Sum.v[i+1]
                Sum.rows = np.delete(Sum.rows, i)
                Sum.cols = np.delete(Sum.cols, i)
                Sum.v = np.delete(Sum.v, i)
                if abs(result) >= self.tol:
                    Sum.v[i] = result
                else:
                    Sum.rows = np.delete(Sum.rows, i)
                    Sum.cols = np.delete(Sum.cols, i)
                    Sum.v = np.delete(Sum.v, i)
                    coords = np.delete(coords, i, axis=0)
                i -= 1

        Sum.number_of_nonzero = len(coords)        
        if Sum.intern_represent == 'CSC':
            Sum.cols = compress(Sum.cols, Sum.num_cols, Sum.number_of_nonzero)
        else:
            Sum.rows = compress(Sum.rows, Sum.num_rows, Sum.number_of_nonzero)
        return Sum
    

    def multiply(self, vector):
        """
        

        Parameters
        ----------
        vector : np.array
            A vector consists of a 1-dimensional numpy array.

        Raises
        ------
        ValueError
            This error is raised when the input to vector multiplication isn't a vector, 
            or when the input vector doesn't match the matrix for multiplication.

        Returns
        -------
        out : np.array
            The output vector resulting from the vector multiplication.

        """
        if len(vector.shape) != 1:
            raise ValueError(f"The input to vector multiplication is not a vector: {vector}")
        vLen = vector.shape[0]
        if vLen != self.num_cols:
            raise ValueError(f"The input vector {vector} does not match the matrix for multiplication.")
        
        out = np.zeros(self.num_rows)
        if self.intern_represent == 'CSR':
            for i in range(self.num_rows):
                start, end = self.rows[i], self.rows[i + 1]
                slLen = end - start
                vals, inds = self.v[start:end], self.cols[start:end]
                for j in range(slLen):
                    out[i] += vals[j] * vector[inds[j]]
        elif self.intern_represent == 'CSC':
            for i in range(self.num_cols):
                start, end = self.cols[i], self.cols[i + 1]
                slLen = end - start
                vals, inds = self.v[start:end], self.rows[start:end]
                vecVal = vector[i]
                for j in range(slLen):
                    out[inds[j]] += vals[j] * vecVal
        else:
            raise ValueError("Unrecognized internal representation for vector multiplication.")
        return out
    
    def describe(self):
        """
        

        Returns
        -------
        None.

        """
        print('==== DESCRIPTION ====')
        print('intern_represent -', self.intern_represent)
        print('V (value) -', self.v)
        print('col_index -', self.cols)
        print('row_index -', self.rows)
        print('Number of rows -', self.num_rows)
        print('Number of columns -', self.num_cols)
        print('Number of nonzero values -', self.number_of_nonzero)
        print('=====================')
        
    def show(self):
        """
        

        Returns
        -------
        None.

        """
        Sum = copy.copy(self)

        if Sum.intern_represent == 'CSR':
            Sum.rows = decompress(Sum.rows, Sum.num_rows)
        else:
            Sum.cols = decompress(Sum.cols, Sum.num_cols)
            Sum.rows, Sum.cols, Sum.v = reorder(Sum.rows, Sum.cols, Sum.v)

        result = np.zeros((Sum.num_rows, Sum.num_cols))
        for i in range(Sum.number_of_nonzero):
            result[Sum.rows[i], Sum.cols[i]]=Sum.v[i]
        print(result)
        

prnt("EXAMPLE")

matrix = np.array([[10,20, 0,  0,  0,  0 ],\
                   [0, 30, 0,  40, 0,  0 ],\
                   [0, 0,  50, 60, 70, 0 ],\
                   [0, 0,  0,  0,  0,  80],])
vector = np.array([5, 2, 7, 10, 6, 3])

sparse_matrix = SparseMatrix(matrix, tol = 10**-5)
sparse_matrix.describe()
sparse_matrix.show()
sparse_matrix.switch('CSC')
sparse_matrix.show()
sparse_matrix.switch('CSR')

prnt("SOME ARRAY EXAMPLES")

print(compress([2, 4, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7], 10, 12))
print(compress([0, 1, 1, 2, 3, 3, 4, 5], 6, 8))
print(compress([0, 0, 0, 0], 5, 4))

print(decompress([0, 0, 0, 1, 1, 2, 4, 4, 12, 12, 12], 10))
print(decompress([0, 1, 3, 4, 6, 7, 8], 6))
print(decompress([0, 4, 4, 4, 4, 4], 5))


prnt("SWITCH TO CSC")

sparse_matrix.switch('CSC')
sparse_matrix.describe()


prnt("THE CHANGE METHOD")

sparse_matrix.change(0, 0, 15)
sparse_matrix.change(1, 2, 35)
sparse_matrix.describe()
sparse_matrix.change(0, 0, 0)
sparse_matrix.describe()

sparse_matrix.switch('CSR')
sparse_matrix.change(2, 1, 45)
sparse_matrix.change(3, 5, 85)
sparse_matrix.describe()
sparse_matrix.change(2, 1, 0)
sparse_matrix.describe()
sparse_matrix.change(1, 0, 0)
sparse_matrix.describe()


prnt("VECTOR MULTIPLICATION")

sparse_mult = SparseMatrix(matrix)
sparse_mult.describe()
print("Input vector:", vector)
print("Desired output vector: ", matrix.dot(vector))
print("Output vector:", sparse_mult.multiply(vector))
sparse_mult.switch('CSC')
print("Output using CSC:", sparse_mult.multiply(vector))


prnt("MATRIX ADDITION")
matrix2 = np.array([[10,20, 0,  0,  0,  -1  ],\
                   [0, 30, 0,  0, 0,  0 ],\
                   [0, 0,  50, 60, 70, 0 ],\
                   [0, 100,  0,  0,  0,  80],])
sparse_matrix2 = SparseMatrix(matrix2, tol = 10**-5)

print("First summand")
sparse_matrix.describe()
print("Second summand")
sparse_matrix2.describe()

sparse_matrix3 = SparseMatrix(matrix + matrix2, tol = 10**-5)
print("Sum of matrices")
(sparse_matrix.add(sparse_matrix2)).describe()
print("Expected output")
sparse_matrix3.describe()

prnt("TOLERANCE ADDITION")
matrix3 = np.array([[1, 0, 1],
                    [0, 0, 0],
                    [0, 2, 0]])
matrix4 = np.array([[-1, 0, 0],
                    [0, 0, 0],
                    [0, -2, 0]])
spMat3 = SparseMatrix(matrix3)
print("First tolerance summand")
spMat3.describe()

spMat4 = SparseMatrix(matrix4)
print("Second tolerance summand")
spMat4.describe()
print("Sum of matrices")
spMat3.add(spMat4).describe()
print("Expected output")
SparseMatrix(matrix3 + matrix4).describe()

prnt("EMPTY ADDITION")
matrix5 = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
spMat5 = SparseMatrix(matrix5)
print("Empty summand")
spMat5.describe()
print("Sum of empty")
spMat5.add(spMat5).describe()

prnt("TOEPLITZ")
toe = SparseMatrix.toeplitz(4)
toe.describe()
toe = SparseMatrix.toeplitz(2)
toe.describe()
toe = SparseMatrix.toeplitz(1)
toe.describe()
toe = SparseMatrix.toeplitz(0)
toe.describe()
SparseMatrix.toeplitz(4).add(SparseMatrix.toeplitz(4)).describe()

prnt("BIGGER TOEPLITZ")
toe = SparseMatrix.toeplitz(10)
toe.describe()
toe = SparseMatrix.toeplitz(100)
toe = SparseMatrix.toeplitz(10000)
toe.describe()

prnt("PERFORMANCE")

toe = SparseMatrix.toeplitz(3)
toe.change(1,0,"nail")
toe.show()
toe.describe()