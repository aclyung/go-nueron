package matrix

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

// Matrix type
type Matrix [][]float64

var errMultiplyIndexNotMatch = errors.New("row of first matrix and colum of second matrix do not match")

// Create 2d Matrix by elements
func Matrix2D(colums [][]float64) (Matrix, error) {
	if colums == nil {
		return nil, errors.New("must have at least 1 parameter")
	}

	mat := Matrix{}
	for _, v := range colums {
		mat = append(mat, v)
	}
	return mat, nil
}

// Create 2d element by shape
func Matrix_shape(rows int, colums int) (Matrix, error) {
	mat := Matrix{}
	for i := 0; i < rows; i++ {
		mat = append(mat, make([]float64, colums))
		for j := 0; j < colums; j++ {
			mat[i][j] = 0
		}
	}
	return mat, nil
}

func Matrix_rand(rows int, colums int) (Matrix, error) {
	mat := Matrix{}
	dist := distuv.Normal{Mu: 0, Sigma: 1}
	for i := 0; i < rows; i++ {
		mat = append(mat, make([]float64, colums))
		for j := 0; j < colums; j++ {
			mat[i][j] = dist.Rand()
		}
	}
	return mat, nil
}

func Sum_row_vector(m Matrix) float64 {
	r := 0.0
	for i := range m {
		r += m[i][0]
	}
	return r
}

func Transpose(m Matrix) Matrix {
	tmat, _ := Matrix_shape(len(m[0]), len(m))
	for i := range tmat {
		for j := range tmat[0] {
			tmat[i][j] = m[j][i]
		}
	}
	return tmat
}

// Matrix Multiplication Function
func Multiply(a Matrix, b Matrix) (Matrix, error) {
	if len(a[0]) != len(b) {
		return nil, errMultiplyIndexNotMatch
	}
	n := len(a[0])
	rows, colums := len(a), len(b[0])
	m, _ := Matrix_shape(rows, colums)
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			for k := 0; k < n; k++ {
				m[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return m, nil
}

// Sigmoid Function
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-1*float64(x)))
}

func Sigmoid_Matrix(m Matrix) Matrix {
	for i := range m {
		for j := range m[i] {
			m[i][j] = sigmoid(m[i][j])
		}
	}
	return m
}

func ErrorFunc() {
}

func Multiply_elem(m1 Matrix, m2 Matrix) Matrix {
	m, _ := Matrix_shape(len(m1), len(m1[0]))
	for i := range m1 {
		for j := range m2 {
			m[i][j] = m1[i][j] * m2[i][j]
		}
	}
	return m
}

func Sum(m1 Matrix, m2 Matrix) Matrix {
	for i := range m1 {
		for j := range m2 {
			m1[i][j] += m2[i][j]
		}
	}
	return m1
}

func Diff(m1 Matrix, m2 Matrix) Matrix {
	for i := range m1 {
		for j := range m2 {
			m1[i][j] -= m2[i][j]
		}
	}
	return m1
}
