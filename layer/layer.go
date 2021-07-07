package layer

import (
	"almeng.com/neuron/matrix"
)

// Layer structure
type Layer struct {
	Weights matrix.Matrix
	Output  int
}

// Create Dense Layer
func Dense(input int, output int) *Layer {
	layer := &Layer{}
	wm, _ := matrix.Matrix_rand(output, input)
	layer.Weights = wm
	layer.Output = output
	return layer
}

func (l Layer) Run(input_Mat matrix.Matrix) matrix.Matrix {
	m, _ := matrix.Multiply(l.Weights, input_Mat)
	return matrix.Sigmoid_Matrix(m)
}
