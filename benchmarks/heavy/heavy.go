// Heavy CPU benchmarks in Go — same algorithms, sizes and random numbers as heavy.apsi. Build: go build heavy.go
package main

import (
	"fmt"
	"math"
	"strconv"
	"time"
)

func rngNext(s int64) int64 { return (s * 16807) % 2147483647 }
func u(s int64) float64     { return float64(s) / 2147483647.0 }

func nbodyDense(n, steps int) float64 {
	x, y, z := make([]float64, n), make([]float64, n), make([]float64, n)
	vx, vy, vz := make([]float64, n), make([]float64, n), make([]float64, n)
	ax, ay, az, m := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
	s := int64(42)
	for i := 0; i < n; i++ {
		s = rngNext(s)
		x[i] = u(s) - 0.5
		s = rngNext(s)
		y[i] = u(s) - 0.5
		s = rngNext(s)
		z[i] = u(s) - 0.5
		s = rngNext(s)
		m[i] = 1.0 / float64(n) * (0.5 + u(s))
	}
	dt, eps2 := 0.001, 0.0001
	for st := 0; st < steps; st++ {
		for i := 0; i < n; i++ {
			fx, fy, fz := 0.0, 0.0, 0.0
			xi, yi, zi := x[i], y[i], z[i]
			for j := 0; j < n; j++ {
				dx, dy, dz := x[j]-xi, y[j]-yi, z[j]-zi
				d2 := dx*dx + dy*dy + dz*dz + eps2
				inv := m[j] / (d2 * math.Sqrt(d2))
				fx += dx * inv
				fy += dy * inv
				fz += dz * inv
			}
			ax[i], ay[i], az[i] = fx, fy, fz
		}
		for i := 0; i < n; i++ {
			vx[i] += dt * ax[i]
			vy[i] += dt * ay[i]
			vz[i] += dt * az[i]
			x[i] += dt * vx[i]
			y[i] += dt * vy[i]
			z[i] += dt * vz[i]
		}
	}
	e := 0.0
	for i := 0; i < n; i++ {
		e += 0.5 * m[i] * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i])
	}
	return e
}

func jacobi(n, iters int) float64 {
	a, b := make([]float64, n*n), make([]float64, n*n)
	for j := 0; j < n; j++ {
		a[j], b[j] = 100.0, 100.0
	}
	for it := 0; it < iters; it++ {
		for i := 1; i < n-1; i++ {
			for j := 1; j < n-1; j++ {
				b[i*n+j] = 0.25 * (a[(i-1)*n+j] + a[(i+1)*n+j] + a[i*n+j-1] + a[i*n+j+1])
			}
		}
		for i := 1; i < n-1; i++ {
			for j := 1; j < n-1; j++ {
				a[i*n+j] = 0.25 * (b[(i-1)*n+j] + b[(i+1)*n+j] + b[i*n+j-1] + b[i*n+j+1])
			}
		}
	}
	s := 0.0
	for _, v := range a {
		s += v
	}
	return s
}

func matmul(n int) float64 {
	a, b, c := make([]float64, n*n), make([]float64, n*n), make([]float64, n*n)
	s := int64(7)
	for i := 0; i < n*n; i++ {
		s = rngNext(s)
		a[i] = u(s) - 0.5
		s = rngNext(s)
		b[i] = u(s) - 0.5
	}
	for i := 0; i < n; i++ {
		for k := 0; k < n; k++ {
			aik := a[i*n+k]
			ci, bk := c[i*n:i*n+n], b[k*n:k*n+n]
			for j := range ci {
				ci[j] += aik * bk[j]
			}
		}
	}
	t := 0.0
	for i := 0; i < n; i++ {
		t += c[i*n+(i*7)%n]
	}
	return t
}

func luLogdet(n int) float64 {
	a := make([]float64, n*n)
	s := int64(11)
	for i := range a {
		s = rngNext(s)
		a[i] = u(s) - 0.5
	}
	logdet := 0.0
	for k := 0; k < n; k++ {
		p, best := k, math.Abs(a[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := math.Abs(a[i*n+k]); v > best {
				best, p = v, i
			}
		}
		if p != k {
			for j := 0; j < n; j++ {
				a[k*n+j], a[p*n+j] = a[p*n+j], a[k*n+j]
			}
		}
		piv := a[k*n+k]
		logdet += math.Log(math.Abs(piv))
		for i := k + 1; i < n; i++ {
			f := a[i*n+k] / piv
			a[i*n+k] = f
			for j := k + 1; j < n; j++ {
				a[i*n+j] -= f * a[k*n+j]
			}
		}
	}
	return logdet
}

func ising(n, sweeps int, beta float64) int64 {
	sp := make([]float64, n*n)
	s := int64(5)
	for i := range sp {
		s = rngNext(s)
		if s < 1073741824 {
			sp[i] = 1.0
		} else {
			sp[i] = -1.0
		}
	}
	w4, w8 := math.Exp(-2.0*beta*4.0), math.Exp(-2.0*beta*8.0)
	for sw := 0; sw < sweeps; sw++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				nb := sp[((i+1)%n)*n+j] + sp[((i+n-1)%n)*n+j] + sp[i*n+(j+1)%n] + sp[i*n+(j+n-1)%n]
				de := 2.0 * sp[i*n+j] * nb
				s = rngNext(s)
				r := u(s)
				if de <= 0.0 || (de == 4.0 && r < w4) || (de == 8.0 && r < w8) {
					sp[i*n+j] = -sp[i*n+j]
				}
			}
		}
	}
	mag := int64(0)
	for _, v := range sp {
		mag += int64(v)
	}
	return mag
}

func levenshtein(n int) int64 {
	a, b := make([]float64, n), make([]float64, n)
	s := int64(3)
	for i := 0; i < n; i++ {
		s = rngNext(s)
		a[i] = float64(s % 4)
		s = rngNext(s)
		b[i] = float64(s % 4)
	}
	prev, cur := make([]float64, n+1), make([]float64, n+1)
	for j := 0; j <= n; j++ {
		prev[j] = float64(j)
	}
	for i := 1; i <= n; i++ {
		cur[0] = float64(i)
		ai := a[i-1]
		for j := 1; j <= n; j++ {
			cost := 1.0
			if ai == b[j-1] {
				cost = 0.0
			}
			cur[j] = math.Min(math.Min(prev[j]+1.0, cur[j-1]+1.0), prev[j-1]+cost)
		}
		copy(prev, cur)
	}
	return int64(prev[n])
}

func vdpRK4(steps int) float64 {
	x, v := 2.0, 0.0
	h, mu := 0.0001, 2.0
	for i := 0; i < steps; i++ {
		k1x, k1v := v, mu*(1.0-x*x)*v-x
		x2, v2 := x+0.5*h*k1x, v+0.5*h*k1v
		k2x, k2v := v2, mu*(1.0-x2*x2)*v2-x2
		x3, v3 := x+0.5*h*k2x, v+0.5*h*k2v
		k3x, k3v := v3, mu*(1.0-x3*x3)*v3-x3
		x4, v4 := x+h*k3x, v+h*k3v
		k4x, k4v := v4, mu*(1.0-x4*x4)*v4-x4
		x += h / 6.0 * (k1x + 2.0*k2x + 2.0*k3x + k4x)
		v += h / 6.0 * (k1v + 2.0*k2v + 2.0*k3v + k4v)
	}
	return x + v
}

func kmeans(n, k, iters int) float64 {
	px, py := make([]float64, n), make([]float64, n)
	cx, cy, sx, sy, cnt := make([]float64, k), make([]float64, k), make([]float64, k), make([]float64, k), make([]float64, k)
	s := int64(9)
	for i := 0; i < n; i++ {
		s = rngNext(s)
		c := s % int64(k)
		s = rngNext(s)
		px[i] = float64(c%4)*3.0 + u(s)
		s = rngNext(s)
		py[i] = float64(c/4)*3.0 + u(s)
	}
	for c := 0; c < k; c++ {
		cx[c], cy[c] = px[c], py[c]
	}
	inertia := 0.0
	for it := 0; it < iters; it++ {
		for c := 0; c < k; c++ {
			sx[c], sy[c], cnt[c] = 0, 0, 0
		}
		inertia = 0
		for i := 0; i < n; i++ {
			bc, bd := 0, 1e300
			for c := 0; c < k; c++ {
				dx, dy := px[i]-cx[c], py[i]-cy[c]
				if d := dx*dx + dy*dy; d < bd {
					bd, bc = d, c
				}
			}
			sx[bc] += px[i]
			sy[bc] += py[i]
			cnt[bc] += 1.0
			inertia += bd
		}
		for c := 0; c < k; c++ {
			if cnt[c] > 0 {
				cx[c], cy[c] = sx[c]/cnt[c], sy[c]/cnt[c]
			}
		}
	}
	return inertia
}

var sink float64

func timeit(name string, f func() float64, isInt bool) {
	r := f()
	best := math.MaxFloat64
	for k := 0; k < 3; k++ {
		t := time.Now()
		sink += f()
		best = math.Min(best, time.Since(t).Seconds())
	}
	chk := strconv.FormatFloat(r, 'g', 12, 64)
	if isInt {
		chk = strconv.FormatInt(int64(r), 10)
	}
	fmt.Printf("%s: %.1f ms  [check %s]\n", name, best*1000, chk)
}

func main() {
	timeit("nbody-dense 2000x10", func() float64 { return nbodyDense(2000, 10) }, false)
	timeit("jacobi-2d 1024x200", func() float64 { return jacobi(1024, 200) }, false)
	timeit("matmul 1024", func() float64 { return matmul(1024) }, false)
	timeit("lu-pivot 800", func() float64 { return luLogdet(800) }, false)
	timeit("ising 512x50", func() float64 { return float64(ising(512, 50, 0.44)) }, true)
	timeit("levenshtein 10k", func() float64 { return float64(levenshtein(10000)) }, true)
	timeit("ode-rk4 20M", func() float64 { return vdpRK4(20000000) }, false)
	timeit("kmeans 200k k16 x20", func() float64 { return kmeans(200000, 16, 20) }, false)
}
