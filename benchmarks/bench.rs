use std::time::Instant;
use std::hint::black_box;
fn fib(n: i64) -> i64 { if n < 2 { n } else { fib(n-1) + fib(n-2) } }
fn mandel_row(y: i64, n: i64, maxit: i64) -> i64 { let mut count=0; let ci=2.0*y as f64/n as f64-1.0;
  for x in 0..n { let cr=2.0*x as f64/n as f64-1.5; let (mut zr,mut zi)=(0.0f64,0.0f64); let mut it=0;
    while it<maxit && zr*zr+zi*zi<=4.0 { let t=zr*zr-zi*zi+cr; zi=2.0*zr*zi+ci; zr=t; it+=1; } if it==maxit {count+=1;} } count }
fn mandelbrot(n: i64, maxit: i64) -> i64 { (0..n).map(|y| mandel_row(y,n,maxit)).sum() }
#[inline] fn a_ij(i: usize, j: usize) -> f64 { 1.0/(((i+j)*(i+j+1)/2+i+1) as f64) }
fn mul_av(v:&[f64],o:&mut [f64]){let n=v.len();for i in 0..n{let mut s=0.0;for j in 0..n{s+=a_ij(i,j)*v[j];}o[i]=s;}}
fn mul_atv(v:&[f64],o:&mut [f64]){let n=v.len();for i in 0..n{let mut s=0.0;for j in 0..n{s+=a_ij(j,i)*v[j];}o[i]=s;}}
fn spectral(n: usize) -> f64 { let mut u=vec![1.0;n]; let mut v=vec![0.0;n]; let mut t=vec![0.0;n];
  for _ in 0..10 { mul_av(&u,&mut t); mul_atv(&t,&mut v); mul_av(&v,&mut t); mul_atv(&t,&mut u); }
  let (mut vbv,mut vv)=(0.0f64,0.0f64); for i in 0..n { vbv+=u[i]*v[i]; vv+=v[i]*v[i]; } (vbv/vv).sqrt() }
fn sieve(n: usize) -> i64 { let mut c=vec![false;n+1]; let mut count=0; for i in 2..=n { if !c[i] { count+=1; let mut j=i*i; while j<=n { c[j]=true; j+=i; } } } count }
fn nbody(steps: i64) -> f64 { let pi4=4.0*std::f64::consts::PI*std::f64::consts::PI; let dp=365.24;
 let mut x:[f64;5]=[0.0,4.84143144246472090e+00,8.34336671824457987e+00,1.28943695621391310e+01,1.53796971148509165e+01];
 let mut y:[f64;5]=[0.0,-1.16032004402742839e+00,4.12479856412430479e+00,-1.51111514016986312e+01,-2.59193146099879641e+01];
 let mut z:[f64;5]=[0.0,-1.03622044471123109e-01,-4.03523417114321381e-01,-2.23307578892655734e-01,1.79258772950371181e-01];
 let mut vx=[0.0,1.66007664274403694e-03*dp,-2.76742510726862411e-03*dp,2.96460137564761618e-03*dp,2.68067772490389322e-03*dp];
 let mut vy=[0.0,7.69901118419740425e-03*dp,4.99852801234917238e-03*dp,2.37847173959480950e-03*dp,1.62824170038242295e-03*dp];
 let mut vz=[0.0,-6.90460016972063023e-05*dp,2.30417297573763929e-05*dp,-2.96589568540237556e-05*dp,-9.51592254519715870e-05*dp];
 let m=[pi4,9.54791938424326609e-04*pi4,2.85885980666130812e-04*pi4,4.36624404335156298e-05*pi4,5.15138902046611451e-05*pi4];
 let (mut px,mut py,mut pz)=(0.0,0.0,0.0); for i in 0..5 { px+=vx[i]*m[i]; py+=vy[i]*m[i]; pz+=vz[i]*m[i]; }
 vx[0]=-px/pi4; vy[0]=-py/pi4; vz[0]=-pz/pi4; let dt=0.01;
 for _ in 0..steps { for i in 0..5 { for j in i+1..5 { let dx=x[i]-x[j]; let dy=y[i]-y[j]; let dz=z[i]-z[j];
   let d2=dx*dx+dy*dy+dz*dz; let mag=dt/(d2*d2.sqrt()); vx[i]-=dx*m[j]*mag; vy[i]-=dy*m[j]*mag; vz[i]-=dz*m[j]*mag;
   vx[j]+=dx*m[i]*mag; vy[j]+=dy*m[i]*mag; vz[j]+=dz*m[i]*mag; } }
   for i in 0..5 { x[i]+=dt*vx[i]; y[i]+=dt*vy[i]; z[i]+=dt*vz[i]; } }
 let mut e=0.0f64; for i in 0..5 { e+=0.5*m[i]*(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]);
   for j in i+1..5 { let dx=x[i]-x[j]; let dy=y[i]-y[j]; let dz=z[i]-z[j]; e-=m[i]*m[j]/(dx*dx+dy*dy+dz*dz).sqrt(); } } e }
fn matmul(n: usize, a:&[f64], b:&[f64], c:&mut [f64]) { for i in 0..n { for k in 0..n { let aik=a[i*n+k]; for j in 0..n { c[i*n+j]+=aik*b[k*n+j]; } } } }
fn hypot_sum(a:&[f64], b:&[f64]) -> f64 { let mut s=0.0f64; for i in 0..a.len() { s+=(a[i]*a[i]+b[i]*b[i]).sqrt(); } s }
fn t<F: FnMut()>(name:&str, mut f:F){ f(); let mut best=1e9f64; for _ in 0..5 { let s=Instant::now(); f(); let d=s.elapsed().as_secs_f64(); if d<best {best=d;} } println!("{}: {:.2} ms", name, best*1000.0); }
fn main(){ let n=512usize; let mut seed=1u64; let mut rnd=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed>>11) as f64)/((1u64<<53) as f64)-0.5};
 let a:Vec<f64>=(0..n*n).map(|_|rnd()).collect(); let b:Vec<f64>=(0..n*n).map(|_|rnd()).collect(); let mut c=vec![0.0;n*n];
 let h1:Vec<f64>=(0..4000000).map(|_|rnd()).collect(); let h2:Vec<f64>=(0..4000000).map(|_|rnd()).collect();
 println!("check {} {} {:.9} {} {:.9}", fib(32), mandelbrot(1000,50), spectral(1000), sieve(10000000), nbody(1000000));
 t("fib(32)",||{black_box(fib(black_box(32)));}); t("mandelbrot 1000x1000 x50",||{black_box(mandelbrot(black_box(1000),black_box(50)));});
 t("spectral-norm 1000",||{black_box(spectral(black_box(1000)));}); t("sieve 10M",||{black_box(sieve(black_box(10000000)));});
 t("nbody 1M steps",||{black_box(nbody(black_box(1000000)));}); t("matmul 512 (naive ikj)",||{matmul(n,&a,&b,&mut c); black_box(&c);});
 t("fused sqrt(a*a+b*b) sum 4M",||{black_box(hypot_sum(black_box(&h1),black_box(&h2)));}); }
