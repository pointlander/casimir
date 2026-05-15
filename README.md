# Neural network n body simulation
## Theory
```
From my clustering algorithm:
A^2*B = B
From the Heisenberg uncertainty principle:
A*B = B*A
Combining the two:
A^2*B = B^2*A
A function that calculates the euclidean distance between
each entry in the matrix and takes the inverse of it is then 
used thus:
A^2*euclidean(B)=B^2*euclidean(A)
Dropout is then used:
dropout(A^2)*euclidean(B)=dropout(B^2)*euclidean(A)
This results in a neural network n body simulation in
two realities: A and B.
```
## Code
```go
euclidean := tf64.B(EuclideanReal)
l0 := tf64.Mul(tf64.Dropout(tf64.Square(n.Set.Get("y")), dropout),
	tf64.Inv(euclidean(n.Set.Get("x"), n.Set.Get("x"))))
loss := tf64.Avg(tf64.Quadratic(tf64.Mul(tf64.Dropout(tf64.Square(n.Set.Get("x")), dropout),
	tf64.Inv(euclidean(n.Set.Get("y"), n.Set.Get("y")))), l0))
```
## Experiments
Below are experiments using different fixed structures in the presence of free particles.
The number of particles to the right of the structure are plotted together for both realities.
## Original
### Simulation
![original simulation](casimir.gif?raw=true)
### Number of particles to the right of the structure
![original plot](dist.png?raw=true)
## Control
### Simulation
![control simulation](control_casimir.gif?raw=true)
### Number of particles to the right of the structure
![control plot](control_dist.png?raw=true)
## Alternate
### Simulation
![alternate simulation](alternate_casimir.gif?raw=true)
### Number of particles to the right of the structure
![alternate plot](alternate_dist.png?raw=true)
## Null
### Simulation
![null simulation](null_casimir.gif?raw=true)
### Number of particles to the right of the null structure
![null plot](null_dist.png?raw=true)
