// Copyright 2026 The Casimir Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/pointlander/gradient/tf64"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-1
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

var palette = []color.Color{}

func init() {
	for i := range 256 {
		g := byte(i)
		palette = append(palette, color.RGBA{g, g, g, 0xff})
	}
}

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
	Images    *gif.GIF
	XYs       plotter.XYs
}

// NewNeuron creates a new neuron
func NewNeuron(seed int64, rows, cols int) Neuron {
	rng := rand.New(rand.NewSource(seed))

	set := tf64.NewSet()
	set.Add("x", 2, rows+10)
	set.Add("y", 2, rows+10)

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
		rng:    rng,
		Set:    &set,
		Images: &gif.GIF{},
		XYs:    make(plotter.XYs, 0, 8),
	}
}

func isDevice(i int) bool {
	return i < 4 || (i >= 8 && i < 10)
}

// Iterate iterates the neuron
func (n *Neuron) Iterate(iterations int) {
	drop := .3
	dropout := map[string]interface{}{
		"rng":  n.rng,
		"drop": &drop,
	}

	{
		x := n.Set.ByName["x"]
		x.X[0] = -1
		x.X[1] = 0 - 2
		x.X[2] = -1
		x.X[3] = 1 - 2
		x.X[4] = -1
		x.X[5] = 2 - 2
		x.X[6] = -1
		x.X[7] = 3 - 2

		/*x.X[8] = 2
		x.X[9] = 0 - 2
		x.X[10] = 2
		x.X[11] = 1 - 2
		x.X[12] = 2
		x.X[13] = 2 - 2
		x.X[14] = 2
		x.X[15] = 3 - 2*/

		x.X[16] = 0
		x.X[17] = 1 - 2
		x.X[18] = 0
		x.X[19] = 2 - 2
	}

	{
		x := n.Set.ByName["y"]
		x.X[0] = -1
		x.X[1] = 0 - 2
		x.X[2] = -1
		x.X[3] = 1 - 2
		x.X[4] = -1
		x.X[5] = 2 - 2
		x.X[6] = -1
		x.X[7] = 3 - 2

		/*x.X[8] = 2
		x.X[9] = 0 - 2
		x.X[10] = 2
		x.X[11] = 1 - 2
		x.X[12] = 2
		x.X[13] = 2 - 2
		x.X[14] = 2
		x.X[15] = 3 - 2*/

		x.X[16] = 0
		x.X[17] = 1 - 2
		x.X[18] = 0
		x.X[19] = 2 - 2
	}

	euclidean := tf64.B(EuclideanReal)

	l0 := tf64.Mul(tf64.Dropout(tf64.Square(n.Set.Get("y")), dropout),
		tf64.Inv(euclidean(n.Set.Get("x"), n.Set.Get("x"))))
	loss := tf64.Avg(tf64.Quadratic(tf64.Mul(tf64.Dropout(tf64.Square(n.Set.Get("x")), dropout),
		tf64.Inv(euclidean(n.Set.Get("y"), n.Set.Get("y")))), l0))

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
				if isDevice(ii / w.S[0]) {
					continue
				}
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
	count := 0
	image := image.NewPaletted(image.Rect(0, 0, 1024, 512), palette)
	{
		x := n.Set.ByName["x"]
		minX, maxX, minY, maxY := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
		for i := range x.S[1] {
			x, y := x.X[i*x.S[0]], x.X[i*x.S[0]+1]
			if x > -1 {
				count++
			}
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
		}
		for i := range x.S[1] {
			xx, yy := x.X[i*x.S[0]], x.X[i*x.S[0]+1]
			x := 500*(xx-minX)/(maxX-minX) + 6
			y := 500*(yy-minY)/(maxY-minY) + 6
			if isDevice(i) {
				for n := -1; n < 2; n++ {
					for m := -1; m < 2; m++ {
						image.Set(n+int(x), m+int(y), color.RGBA{0x55, 0x55, 0x55, 0xff})
					}
				}
				continue
			}
			for n := -1; n < 2; n++ {
				for m := -1; m < 2; m++ {
					image.Set(n+int(x), m+int(y), color.RGBA{0xff, 0xff, 0xff, 0xff})
				}
			}
		}
	}
	{
		for i := range 512 {
			image.Set(512, int(i), color.RGBA{0xff, 0xff, 0xff, 0xff})
		}
	}
	{
		x := n.Set.ByName["y"]
		minX, maxX, minY, maxY := math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
		for i := range x.S[1] {
			x, y := x.X[i*x.S[0]], x.X[i*x.S[0]+1]
			if x > -1 {
				count++
			}
			if x < minX {
				minX = x
			}
			if x > maxX {
				maxX = x
			}
			if y < minY {
				minY = y
			}
			if y > maxY {
				maxY = y
			}
		}
		for i := range x.S[1] {
			xx, yy := x.X[i*x.S[0]], x.X[i*x.S[0]+1]
			x := 500*(xx-minX)/(maxX-minX) + 6
			y := 500*(yy-minY)/(maxY-minY) + 6
			if isDevice(i) {
				for n := -1; n < 2; n++ {
					for m := -1; m < 2; m++ {

						image.Set(n+int(x)+512, m+int(y), color.RGBA{0x55, 0x55, 0x55, 0xff})
					}
				}
				continue
			}
			for n := -1; n < 2; n++ {
				for m := -1; m < 2; m++ {

					image.Set(n+int(x)+512, m+int(y), color.RGBA{0xff, 0xff, 0xff, 0xff})
				}
			}
		}
	}
	{
		for i := range 512 {
			for ii := range 4 {
				image.Set(int(float64(n.Iteration*i)/float64(512)), 511-ii, color.RGBA{0xff, 0xff, 0xff, 0xff})
			}
		}
	}
	n.Images.Image = append(n.Images.Image, image)
	n.Images.Delay = append(n.Images.Delay, 10)
	n.XYs = append(n.XYs, plotter.XY{X: float64(n.Iteration), Y: float64(count)})
}

func main() {
	neuron := NewNeuron(1, 33, 33)
	for range 1024 {
		neuron.Iterate(1)
	}

	{
		out, err := os.Create("casimir.gif")
		if err != nil {
			panic(err)
		}
		defer out.Close()
		err = gif.EncodeAll(out, neuron.Images)
		if err != nil {
			panic(err)
		}
	}
	{
		p := plot.New()

		p.Title.Text = "count vs iteration"
		p.X.Label.Text = "iteration"
		p.Y.Label.Text = "count"

		scatter, err := plotter.NewScatter(neuron.XYs)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "dist.png")
		if err != nil {
			panic(err)
		}
	}
}
