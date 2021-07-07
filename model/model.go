package model

import (
	"fmt"

	"almeng.com/neuron/layer"
	"almeng.com/neuron/matrix"
)

// model structure
type model struct {
	layers []layer.Layer
}

// Initialize model and Input node
func Init(outunit int, input int) *model {
	m := &model{}
	m.layers = append(m.layers, *layer.Dense(input, outunit))
	return m
}

// Add layers to Model
func (m *model) Add(outunit int) {
	input_dim := m.layers[len(m.layers)-1].Output
	m.layers = append(m.layers, *layer.Dense(input_dim, outunit))
}

// Run model
func (n *model) Run(input_Mat []matrix.Matrix, target_values []matrix.Matrix) {
	for a, i := range input_Mat {
		//fmt.Println(i)
		var res matrix.Matrix
		in := matrix.Transpose(i)
		t := matrix.Transpose(target_values[a])
		for b, v := range n.layers {
			fmt.Printf("[%d][%d]:", a, b)
			fmt.Print(in, ":", len(in), "/")
			out := v.Run(in)
			in = out
			fmt.Println(in, len(in))
			res = out
		}
		err := matrix.Diff(t, res)
		for b, _ := range n.layers {
			l := &n.layers[len(n.layers)-b-1]
			er, _ := matrix.Multiply(l.Weights, err)
			l.Weights = matrix.Sum(&l.Weights, matrix.Multiply_elem())

		}
		fmt.Println(res, "error:", err)
	}
}
