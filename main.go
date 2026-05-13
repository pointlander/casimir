// Copyright 2026 The Casimir Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Euclidean computes the euclidean distance between all row vectors and all row vectors
func EuclideanReal(k tf64.Continuation, node int, a, b *tf64.V, options ...map[string]interface{}) bool {
	if len(a.S) != 2 || len(b.S) != 2 {
		panic("tensor needs to have two dimensions")
	}
	width := a.S[0]
	if width != b.S[0] || a.S[1] != b.S[1] {
		panic("dimensions are not the same")
	}
	c, sizeA, sizeB := tf64.NewV(a.S[1], b.S[1]), len(a.X), len(b.X)
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, sum := a.X[i:i+width], b.X[ii:ii+width], 0.0
			for j, ax := range av {
				diff := (ax - bv[j])
				sum += diff * diff
			}
			c.X = append(c.X, math.Sqrt(sum))
		}
	}
	if k(&c) {
		return true
	}
	index := 0
	for i := 0; i < sizeA; i += width {
		for ii := 0; ii < sizeB; ii += width {
			av, bv, cx, ad, bd, d := a.X[i:i+width], b.X[ii:ii+width], c.X[index], a.D[i:i+width], b.D[ii:ii+width], c.D[index]
			for j, ax := range av {
				if cx == 0 {
					continue
				}
				ad[j] += (ax - bv[j]) * d / cx
				bd[j] += (bv[j] - ax) * d / cx
			}
			index++
		}
	}
	return false
}

// Neuron is a neuromorphic neuron
type Neuron struct {
	Iteration int
	Set       *tf64.Set
	rng       *rand.Rand
}

// NewNeuron creates a new neuron
func NewNeuron(seed int64, rows, cols int) Neuron {
	rng := rand.New(rand.NewSource(seed))

	set := tf64.NewSet()
	set.Add("x", 2, rows)

	for ii := range set.Weights {
		w := set.Weights[ii]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	return Neuron{
		rng: rng,
		Set: &set,
	}
}

// Iterate iterates the neuron
func (n *Neuron) Iterate(iterations int, Eta float64, y *tf64.Set) {
	drop := .3
	dropout := map[string]interface{}{
		"rng":  n.rng,
		"drop": &drop,
	}

	euclidean := tf64.B(EuclideanReal)

	l0 := tf64.Mul(tf64.Dropout(tf64.Square(y.Get("x")), dropout),
		tf64.Inv(euclidean(n.Set.Get("x"), n.Set.Get("x"))))
	loss := tf64.Avg(tf64.Quadratic(tf64.Mul(tf64.Dropout(tf64.Square(n.Set.Get("x")), dropout),
		tf64.Inv(euclidean(y.Get("x"), y.Get("x")))), l0))

	var l float64
	for range iterations {
		iteration := n.Iteration
		pow := func(x float64) float64 {
			y := math.Pow(x, float64(iteration+1))
			if math.IsNaN(y) || math.IsInf(y, 0) {
				return 0
			}
			return y
		}

		n.Set.Zero()
		y.Zero()
		l = tf64.Gradient(loss).X[0]
		if math.IsNaN(l) || math.IsInf(l, 0) {
			fmt.Println(iteration, l)
			return
		}

		norm := 0.0
		for _, p := range n.Set.Weights {
			for _, d := range p.D {
				norm += d * d
			}
		}
		norm = math.Sqrt(norm)
		b1, b2 := pow(B1), pow(B2)
		scaling := 1.0
		if norm > 1 {
			scaling = 1 / norm
		}
		for _, w := range n.Set.Weights {
			for ii, d := range w.D {
				g := d * scaling
				m := B1*w.States[StateM][ii] + (1-B1)*g
				v := B2*w.States[StateV][ii] + (1-B2)*g*g
				w.States[StateM][ii] = m
				w.States[StateV][ii] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				if vhat < 0 {
					vhat = 0
				}
				w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
			}
		}
		n.Iteration++
	}
	//fmt.Println(l)
}

func main() {

}
