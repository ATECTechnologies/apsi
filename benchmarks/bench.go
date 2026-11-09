package main
import ("fmt";"math";"time")
func fib(n int64) int64 { if n < 2 { return n }; return fib(n-1)+fib(n-2) }
func mandelRow(y,n,maxit int64) int64 { var count int64; ci:=2.0*float64(y)/float64(n)-1.0
 for x:=int64(0);x<n;x++ { cr:=2.0*float64(x)/float64(n)-1.5; zr,zi:=0.0,0.0; it:=int64(0)
  for it<maxit && zr*zr+zi*zi<=4.0 { t:=zr*zr-zi*zi+cr; zi=2.0*zr*zi+ci; zr=t; it++ }; if it==maxit {count++} }; return count }
func mandelbrot(n,maxit int64) int64 { var t int64; for y:=int64(0);y<n;y++ { t+=mandelRow(y,n,maxit) }; return t }
func aij(i,j int) float64 { return 1.0/float64((i+j)*(i+j+1)/2+i+1) }
func mulAv(v,o []float64){n:=len(v);for i:=0;i<n;i++{s:=0.0;for j:=0;j<n;j++{s+=aij(i,j)*v[j]};o[i]=s}}
func mulAtv(v,o []float64){n:=len(v);for i:=0;i<n;i++{s:=0.0;for j:=0;j<n;j++{s+=aij(j,i)*v[j]};o[i]=s}}
func spectral(n int) float64 { u:=make([]float64,n); v:=make([]float64,n); t:=make([]float64,n); for i:=range u {u[i]=1}
 for k:=0;k<10;k++ { mulAv(u,t); mulAtv(t,v); mulAv(v,t); mulAtv(t,u) }; vbv,vv:=0.0,0.0; for i:=0;i<n;i++ {vbv+=u[i]*v[i]; vv+=v[i]*v[i]}; return math.Sqrt(vbv/vv) }
func sieve(n int) int64 { c:=make([]bool,n+1); var count int64; for i:=2;i<=n;i++ { if !c[i] { count++; for j:=i*i;j<=n;j+=i {c[j]=true} } }; return count }
func nbody(steps int) float64 { pi4:=4*math.Pi*math.Pi; dp:=365.24
 x:=[5]float64{0,4.84143144246472090e+00,8.34336671824457987e+00,1.28943695621391310e+01,1.53796971148509165e+01}
 y:=[5]float64{0,-1.16032004402742839e+00,4.12479856412430479e+00,-1.51111514016986312e+01,-2.59193146099879641e+01}
 z:=[5]float64{0,-1.03622044471123109e-01,-4.03523417114321381e-01,-2.23307578892655734e-01,1.79258772950371181e-01}
 vx:=[5]float64{0,1.66007664274403694e-03*dp,-2.76742510726862411e-03*dp,2.96460137564761618e-03*dp,2.68067772490389322e-03*dp}
 vy:=[5]float64{0,7.69901118419740425e-03*dp,4.99852801234917238e-03*dp,2.37847173959480950e-03*dp,1.62824170038242295e-03*dp}
 vz:=[5]float64{0,-6.90460016972063023e-05*dp,2.30417297573763929e-05*dp,-2.96589568540237556e-05*dp,-9.51592254519715870e-05*dp}
 m:=[5]float64{pi4,9.54791938424326609e-04*pi4,2.85885980666130812e-04*pi4,4.36624404335156298e-05*pi4,5.15138902046611451e-05*pi4}
 px,py,pz:=0.0,0.0,0.0; for i:=0;i<5;i++ {px+=vx[i]*m[i];py+=vy[i]*m[i];pz+=vz[i]*m[i]}; vx[0]=-px/pi4;vy[0]=-py/pi4;vz[0]=-pz/pi4; dt:=0.01
 for s:=0;s<steps;s++ { for i:=0;i<5;i++ { for j:=i+1;j<5;j++ { dx:=x[i]-x[j];dy:=y[i]-y[j];dz:=z[i]-z[j]; d2:=dx*dx+dy*dy+dz*dz; mag:=dt/(d2*math.Sqrt(d2))
   vx[i]-=dx*m[j]*mag;vy[i]-=dy*m[j]*mag;vz[i]-=dz*m[j]*mag;vx[j]+=dx*m[i]*mag;vy[j]+=dy*m[i]*mag;vz[j]+=dz*m[i]*mag } }
  for i:=0;i<5;i++ {x[i]+=dt*vx[i];y[i]+=dt*vy[i];z[i]+=dt*vz[i]} }
 e:=0.0; for i:=0;i<5;i++ { e+=0.5*m[i]*(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]); for j:=i+1;j<5;j++ {dx:=x[i]-x[j];dy:=y[i]-y[j];dz:=z[i]-z[j]; e-=m[i]*m[j]/math.Sqrt(dx*dx+dy*dy+dz*dz)} }; return e }
func matmul(n int,a,b,c []float64){for i:=0;i<n;i++{for k:=0;k<n;k++{aik:=a[i*n+k];for j:=0;j<n;j++{c[i*n+j]+=aik*b[k*n+j]}}}}
func hypotSum(a,b []float64) float64 { s:=0.0; for i:=range a { s+=math.Sqrt(a[i]*a[i]+b[i]*b[i]) }; return s }
var sink float64
func t(name string, f func()) { f(); best:=1e9; for r:=0;r<5;r++ { s:=time.Now(); f(); d:=time.Since(s).Seconds(); if d<best {best=d} }; fmt.Printf("%s: %.2f ms\n", name, best*1000) }
func main(){ n:=512; seed:=uint64(1); rnd:=func() float64 { seed=seed*6364136223846793005+1442695040888963407; return float64(seed>>11)/float64(uint64(1)<<53)-0.5 }
 a:=make([]float64,n*n); b:=make([]float64,n*n); c:=make([]float64,n*n); for i:=range a {a[i]=rnd(); b[i]=rnd()}
 h1:=make([]float64,4000000); h2:=make([]float64,4000000); for i:=range h1 {h1[i]=rnd(); h2[i]=rnd()}
 fmt.Printf("check %d %d %.9f %d %.9f\n", fib(32), mandelbrot(1000,50), spectral(1000), sieve(10000000), nbody(1000000))
 t("fib(32)",func(){sink+=float64(fib(32))}); t("mandelbrot 1000x1000 x50",func(){sink+=float64(mandelbrot(1000,50))})
 t("spectral-norm 1000",func(){sink+=spectral(1000)}); t("sieve 10M",func(){sink+=float64(sieve(10000000))})
 t("nbody 1M steps",func(){sink+=nbody(1000000)}); t("matmul 512 (naive ikj)",func(){matmul(n,a,b,c)})
 t("fused sqrt(a*a+b*b) sum 4M",func(){sink+=hypotSum(h1,h2)}) }
