package main

import (
	"almeng.com/neuron/matrix"
	"almeng.com/neuron/model"
)

func main() {

	// fmt.Println(r)
	mod := model.Init(3, 3, 0.1)
	mod.Add(3)
	mod.Add(3)
	mod.Add(1)
	m := make([]matrix.Matrix, 1000)
	mat := make([]matrix.Matrix, 1000)
	for i := range m {
		ml, _ := matrix.Matrix2D([][]float64{{0.5, 0.1, 0.8}})
		mt, _ := matrix.Matrix2D([][]float64{{1}})
		m[i] = ml
		mat[i] = mt
	}
	mod.Run(m, mat)
	//m[0] = matrix.Transpose(m[0])

}
