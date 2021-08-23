package layer

import (
	"almeng.com/neuron/matrix"
)

// Layer structure
type Layer struct {
	Weights    matrix.Matrix
	Activation func()

	Output int
}

// Create Dense Layer
func Dense(input int, output int) *Layer {
	layer := &Layer{}
	wm := matrix.Matrix_rand(output, input)
	layer.Weights = wm
	layer.Output = output
	return layer
}

func (l *Layer) UpdateWeight(lr float64, cur_layer_output matrix.Matrix, next_layer_output matrix.Matrix, err matrix.Matrix) {
	// fmt.Println()
	// fmt.Println(len(next_layer_output), len(next_layer_output[0]))
	v1, _ := matrix.Multiply_elem(err, next_layer_output)
	v2 := matrix.RealDiff(1.0, next_layer_output)
	v3, _ := matrix.Multiply_elem(v1, v2)
	v4, _ := matrix.Multiply(v3, matrix.Transpose(cur_layer_output))
	updateWeightval := matrix.Matrix_real_mul(v4, lr)
	l.Weights, _ = matrix.Sum(l.Weights, updateWeightval)

}

func (l Layer) Run(input_Mat matrix.Matrix) matrix.Matrix {
	m, _ := matrix.Multiply(l.Weights, input_Mat)
	return matrix.Sigmoid_Matrix(m)
}
