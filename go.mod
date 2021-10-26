module almeng.com/neuron

require (
	almeng.com/neuron/matrix v0.0.0
	almeng.com/neuron/model v0.0.0
)

replace (
	almeng.com/neuron/layer => ./layer
	almeng.com/neuron/matrix => ./matrix
	almeng.com/neuron/model => ./model
)

go 1.16
