package model

import (
	"fmt"

	"almeng.com/neuron/layer"
	"almeng.com/neuron/matrix"
)

// model structure
type model struct {
	layers        []layer.Layer
	learning_rate float64
}

// Initialize model, Input node and learning rate
func Init(outunit int, input int, lr float64) *model {
	m := &model{}
	m.learning_rate = lr
	l := *layer.Dense(input, outunit)
	l.Weights = matrix.Matrix_fill(l.Weights, 1.0)
	m.layers = append(m.layers, l)
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
		var res []matrix.Matrix
		input := i.Transpose()
		target := target_values[a].Transpose()
		for b, v := range n.layers {
			fmt.Printf("[%d][%d]:", a, b)
			fmt.Print(input, ":", len(input), "/")
			out := v.Run(input)
			input = out
			fmt.Println(input, len(input))
			res = append(res, out)
		}

		var next_layer_error matrix.Matrix
		var next_layer_output matrix.Matrix
		for b := range n.layers {
			if b == 0 {
				next_layer_error, _ = matrix.Diff(target, res[len(res)-1])
				next_layer_output = res[len(res)-1-b]
				continue
			}
			cur_layer_output := res[len(res)-b-1]
			l := &n.layers[len(n.layers)-b]
			err, _ := matrix.Multiply(l.Weights.Transpose(), next_layer_error)
			l.UpdateWeight(n.learning_rate, cur_layer_output, next_layer_output, next_layer_error)
			next_layer_output = cur_layer_output
			next_layer_error = err
			if b == len(n.layers) {
				break
			}
		}
	}
}
