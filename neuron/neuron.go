package neuron

import (
	"almeng.com/neuron/layer"
	_ "almeng.com/neuron/matrix"
)

type neuron struct {
	layers []layer.Layer
}

func Init(outunit int, input int) *neuron {
	n := &neuron{}
	n.layers = append(n.layers, *layer.Dense(outunit, input))
	return n
}

func (n *neuron) Add(outunit int) {
	input_dim := n.layers[len(n.layers)-1].Output
	n.layers = append(n.layers, *layer.Dense(outunit, input_dim))
}

// func (n *neuron) Run() {
// 	for _, layer :=  range n.layers {
// 		layer.
// 	}
// }
