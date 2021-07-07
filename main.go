package main

import (
	"almeng.com/neuron/matrix"
	"almeng.com/neuron/model"
)

func main() {
	// // b, _ := matrix.Matrix2D([][]int{{1, 2}, {3, 4}})
	// // 2x3 matrix
	// c, _ := matrix.Matrix2D([][]float64{{1.0, 2.0, 3.0}, {2.0, 3.0, 4.0}})
	// // 3x5 matrix
	// s, _ := matrix.Matrix2D([][]float64{{1.0, 2.0, 3.0, 4.0, 5.0}, {2.0, 3.0, 4.0, 5.0, 6.0}, {3.0, 4.0, 5.0, 6.0, 7.0}})
	// // muliplycation
	// r, _ := matrix.Multiply(c, s)
	// // model initializing input nodes = 2, output nodes = 2
	// a := model.Init(2, 2)
	// // // add Dense Layer to model input nodes = 2, output nodes = 3
	// a.Add(3)
	// a.Add(4)
	// m := make([]matrix.Matrix, 2)
	// m[0], _ = matrix.Matrix2D([][]float64{{1.0, 3.0}})
	// m[1], _ = matrix.Matrix2D([][]float64{{1.0, 2.0}})
	// t := make([]matrix.Matrix, 2)
	// t[0], _ = matrix.Matrix2D([][]float64{{1.0, 3.0}})
	// t[1], _ = matrix.Matrix2D([][]float64{{1.0, 2.0}})
	// a.Run(m, t)

	// fmt.Println(a)
	// fmt.Println(c)
	// fmt.Println(s)
	// fmt.Println(r)
	mod := model.Init(3, 3)
	mod.Add(3)
	mod.Add(3)
	mod.Add(1)
	m := make([]matrix.Matrix, 1)
	m[0], _ = matrix.Matrix2D([][]float64{{0.9, 0.1, 0.8}})
	mat, _ := matrix.Matrix2D([][]float64{{1}})
	mod.Run(m, []matrix.Matrix{mat})
	//m[0] = matrix.Transpose(m[0])

}
