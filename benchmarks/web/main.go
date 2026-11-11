package main

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

// Minimal keep-alive HTTP/1.1 load generator: N connections, each sends requests back to back.
func main() {
	addr := os.Args[1]
	path := os.Args[2]
	conns, _ := strconv.Atoi(os.Args[3])
	secs, _ := strconv.Atoi(os.Args[4])
	req := []byte("GET " + path + " HTTP/1.1\r\nHost: x\r\n\r\n")
	var count, errs int64
	var lat []int64
	var mu sync.Mutex
	deadline := time.Now().Add(time.Duration(secs) * time.Second)
	var wg sync.WaitGroup
	for i := 0; i < conns; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := net.Dial("tcp", addr)
			if err != nil {
				atomic.AddInt64(&errs, 1)
				return
			}
			r := bufio.NewReaderSize(c, 65536)
			local := make([]int64, 0, 100000)
			for time.Now().Before(deadline) {
				t := time.Now()
				c.Write(req)
				n := -1
				for {
					line, err := r.ReadString('\n')
					if err != nil {
						atomic.AddInt64(&errs, 1)
						return
					}
					if len(line) > 16 && (line[:15] == "Content-Length:" || line[:15] == "content-length:") {
						n, _ = strconv.Atoi(line[16 : len(line)-2])
					}
					if line == "\r\n" {
						break
					}
				}
				buf := make([]byte, n)
				if _, err := io_ReadFull(r, buf); err != nil {
					atomic.AddInt64(&errs, 1)
					return
				}
				local = append(local, time.Since(t).Microseconds())
				atomic.AddInt64(&count, 1)
			}
			mu.Lock()
			lat = append(lat, local...)
			mu.Unlock()
		}()
	}
	wg.Wait()
	sortInts(lat)
	p := func(q float64) int64 {
		if len(lat) == 0 {
			return 0
		}
		return lat[int(q*float64(len(lat)-1))]
	}
	fmt.Printf("%d req in %ds = %.0f req/s, p50 %d us, p99 %d us, errors %d\n", count, secs, float64(count)/float64(secs), p(0.5), p(0.99), errs)
}

func io_ReadFull(r *bufio.Reader, b []byte) (int, error) {
	n := 0
	for n < len(b) {
		m, err := r.Read(b[n:])
		n += m
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

func sortInts(a []int64) {
	if len(a) < 2 {
		return
	}
	// simple radix-free sort
	quick(a, 0, len(a)-1)
}

func quick(a []int64, lo, hi int) {
	for lo < hi {
		p := a[(lo+hi)/2]
		i, j := lo, hi
		for i <= j {
			for a[i] < p {
				i++
			}
			for a[j] > p {
				j--
			}
			if i <= j {
				a[i], a[j] = a[j], a[i]
				i++
				j--
			}
		}
		if j-lo < hi-i {
			quick(a, lo, j)
			lo = i
		} else {
			quick(a, i, hi)
			hi = j
		}
	}
}
