package matrix

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

// Matrix type
type Matrix [][]float64

var ErrMultiplyIndexNotMatch = errors.New("row of first matrix and colum of second matrix do not match")
var ErrMatrixSizeNotMatch = errors.New("both matrices must be the same size")

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

func precision2(f float64) float64 {
	return float64(int(f*100000)) / 100000
}

// Create 2d element by shape
func Matrix_shape(rows int, colums int) Matrix {
	mat := Matrix{}
	for i := 0; i < rows; i++ {
		mat = append(mat, make([]float64, colums))
		for j := 0; j < colums; j++ {
			mat[i][j] = 0
		}
	}
	return mat
}

func Matrix_fill(m Matrix, r float64) Matrix {
	for i := range m {
		for j := range m[0] {
			m[i][j] = r
		}
	}
	return m
}

func Matrix_rand(rows int, colums int) Matrix {
	mat := Matrix{}
	dist := distuv.Normal{Mu: 0, Sigma: 1}
	for i := 0; i < rows; i++ {
		mat = append(mat, make([]float64, colums))
		for j := 0; j < colums; j++ {
			mat[i][j] = precision2(dist.Rand())
		}
	}
	fmt.Println(mat)
	return mat
}

func Sum_row_vector(m Matrix) float64 {
	r := 0.0
	for i := range m {
		r += m[i][0]
	}
	return r
}

func Transpose(m Matrix) Matrix {
	tmat := Matrix_shape(len(m[0]), len(m))
	for i := range tmat {
		for j := range tmat[0] {
			tmat[i][j] = m[j][i]
		}
	}
	return tmat
}

// Multiply Returns Matrix Multiplication Product
func Multiply(a Matrix, b Matrix) (Matrix, error) {
	if len(a[0]) != len(b) {
		return nil, ErrMultiplyIndexNotMatch
	}
	n := len(a[0])
	rows, colums := len(a), len(b[0])
	m := Matrix_shape(rows, colums)
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			for k := 0; k < n; k++ {
				m[i][j] += precision2(a[i][k] * b[k][j])
			}
		}
	}
	return m, nil
}

// Multiply_elem Returns Hadamard product
func Multiply_elem(a Matrix, b Matrix) (m Matrix, e error) {
	if e = SizeCheck(a, b); e != nil {
		return nil, e
	}
	m = Matrix_shape(len(a), len(a[0]))
	for i := range a {
		for j := range a[0] {
			m[i][j] = precision2(a[i][j] * b[i][j])
		}
	}
	return
}

func SizeCheck(a Matrix, b Matrix) error {
	if (len(a) != len(b)) || (len(a[0]) != len(b[0])) {
		return ErrMatrixSizeNotMatch
	}
	return nil
}

// Sigmoid Function
func sigmoid(x float64) float64 {
	return precision2(1 / (1 + math.Exp(-1*float64(x))))
}

func Sigmoid_Matrix(m Matrix) Matrix {
	for i := range m {
		for j := range m[i] {
			m[i][j] = precision2(sigmoid(m[i][j]))
		}
	}
	return m
}

func Sigmoid_Backward() {}

func ReLU(x float64) float64 {
	r := 0.0
	if x > 0 {
		r = x
	}
	return precision2(r)
}

func ReLU_Matrix(m Matrix) Matrix {
	for i := range m {
		for j := range m[i] {
			m[i][j] = precision2(ReLU(m[i][j]))
		}
	}
	return m
}

func Matrix_real_mul(m Matrix, r float64) Matrix {
	for i := range m {
		for j := range m[0] {
			m[i][j] = precision2(m[i][j] * r)
		}
	}
	return m
}

// Sum Returns Sum of two Matrixs
func Sum(m1 Matrix, m2 Matrix) (Matrix, error) {
	if SizeCheck(m1, m2) != nil {
		return nil, ErrMatrixSizeNotMatch
	}
	for i := range m1 {
		for j := range m2 {
			m1[i][j] += m2[i][j]
		}
	}
	return m1, nil
}

// Diffs the value of Matrix by real number and return
func RealDiff(r float64, m Matrix) Matrix {
	mat := Matrix_shape(len(m), len(m[0]))
	mat = Matrix_fill(mat, r)
	return mat
}

func Diff(m1 Matrix, m2 Matrix) (Matrix, error) {
	if SizeCheck(m1, m2) != nil {
		return nil, ErrMatrixSizeNotMatch
	}
	for i := range m1 {
		for j := range m2 {
			m1[i][j] -= m2[i][j]
		}
	}
	return m1, nil
}
