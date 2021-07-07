package neuron

import (
	"math"

	"almeng.com/neuron/layer"
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

func sigmoid(x  int) float64 {
	return 1/(1+math.Exp(-1*float64(x)))
}

func (n *neuron) Run() {
	for _, layer :=  range n.layers {
		layer.
	}
}