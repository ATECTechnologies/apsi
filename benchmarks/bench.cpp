#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
long fib(long n){return n<2?n:fib(n-1)+fib(n-2);}
long mandel_row(long y,long n,long maxit){long count=0;double ci=2.0*y/n-1.0;
 for(long x=0;x<n;x++){double cr=2.0*x/n-1.5,zr=0,zi=0;long it=0;
  while(it<maxit&&zr*zr+zi*zi<=4.0){double t=zr*zr-zi*zi+cr;zi=2.0*zr*zi+ci;zr=t;it++;}
  if(it==maxit)count++;}return count;}
long mandelbrot(long n,long maxit){long t=0;for(long y=0;y<n;y++)t+=mandel_row(y,n,maxit);return t;}
static inline double a_ij(long i,long j){return 1.0/((i+j)*(i+j+1)/2+i+1);}
void mul_av(long n,const double*v,double*o){for(long i=0;i<n;i++){double s=0;for(long j=0;j<n;j++)s+=a_ij(i,j)*v[j];o[i]=s;}}
void mul_atv(long n,const double*v,double*o){for(long i=0;i<n;i++){double s=0;for(long j=0;j<n;j++)s+=a_ij(j,i)*v[j];o[i]=s;}}
double spectral(long n){double*u=(double*)malloc(n*8),*v=(double*)calloc(n,8),*t=(double*)calloc(n,8);for(long i=0;i<n;i++)u[i]=1;
 for(int k=0;k<10;k++){mul_av(n,u,t);mul_atv(n,t,v);mul_av(n,v,t);mul_atv(n,t,u);}
 double vbv=0,vv=0;for(long i=0;i<n;i++){vbv+=u[i]*v[i];vv+=v[i]*v[i];}free(u);free(v);free(t);return sqrt(vbv/vv);}
long sieve(long n){char*c=(char*)calloc(n+1,1);long count=0;for(long i=2;i<=n;i++)if(!c[i]){count++;for(long j=i*i;j<=n;j+=i)c[j]=1;}free(c);return count;}
double nbody(long steps){double pi4=4*M_PI*M_PI,dp=365.24;
 double x[5]={0,4.84143144246472090e+00,8.34336671824457987e+00,1.28943695621391310e+01,1.53796971148509165e+01};
 double y[5]={0,-1.16032004402742839e+00,4.12479856412430479e+00,-1.51111514016986312e+01,-2.59193146099879641e+01};
 double z[5]={0,-1.03622044471123109e-01,-4.03523417114321381e-01,-2.23307578892655734e-01,1.79258772950371181e-01};
 double vx[5]={0,1.66007664274403694e-03*dp,-2.76742510726862411e-03*dp,2.96460137564761618e-03*dp,2.68067772490389322e-03*dp};
 double vy[5]={0,7.69901118419740425e-03*dp,4.99852801234917238e-03*dp,2.37847173959480950e-03*dp,1.62824170038242295e-03*dp};
 double vz[5]={0,-6.90460016972063023e-05*dp,2.30417297573763929e-05*dp,-2.96589568540237556e-05*dp,-9.51592254519715870e-05*dp};
 double m[5]={pi4,9.54791938424326609e-04*pi4,2.85885980666130812e-04*pi4,4.36624404335156298e-05*pi4,5.15138902046611451e-05*pi4};
 double px=0,py=0,pz=0;for(int i=0;i<5;i++){px+=vx[i]*m[i];py+=vy[i]*m[i];pz+=vz[i]*m[i];}
 vx[0]=-px/pi4;vy[0]=-py/pi4;vz[0]=-pz/pi4;double dt=0.01;
 for(long s=0;s<steps;s++){for(int i=0;i<5;i++)for(int j=i+1;j<5;j++){double dx=x[i]-x[j],dy=y[i]-y[j],dz=z[i]-z[j];
  double d2=dx*dx+dy*dy+dz*dz;double mag=dt/(d2*sqrt(d2));vx[i]-=dx*m[j]*mag;vy[i]-=dy*m[j]*mag;vz[i]-=dz*m[j]*mag;
  vx[j]+=dx*m[i]*mag;vy[j]+=dy*m[i]*mag;vz[j]+=dz*m[i]*mag;}
  for(int i=0;i<5;i++){x[i]+=dt*vx[i];y[i]+=dt*vy[i];z[i]+=dt*vz[i];}}
 double e=0;for(int i=0;i<5;i++){e+=0.5*m[i]*(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]);
  for(int j=i+1;j<5;j++){double dx=x[i]-x[j],dy=y[i]-y[j],dz=z[i]-z[j];e-=m[i]*m[j]/sqrt(dx*dx+dy*dy+dz*dz);}}return e;}
void matmul(long n,const double*a,const double*b,double*c){for(long i=0;i<n;i++)for(long k=0;k<n;k++){double aik=a[i*n+k];for(long j=0;j<n;j++)c[i*n+j]+=aik*b[k*n+j];}}
double hypot_sum(long n,const double*a,const double*b){double s=0;for(long i=0;i<n;i++)s+=sqrt(a[i]*a[i]+b[i]*b[i]);return s;}
#define T(name,expr) {expr; double best=1e9; for(int r=0;r<5;r++){double t=now(); expr; double d=now()-t; if(d<best)best=d;} printf("%s: %.2f ms\n",name,best*1000);}
volatile long sinkl; volatile double sinkd; volatile long V32=32,V1000=1000,V50=50,V10M=10000000,V1M=1000000;
int main(){long N=512;double*A=(double*)malloc(N*N*8),*B=(double*)malloc(N*N*8),*C=(double*)calloc(N*N,8);srand(1);
 for(long i=0;i<N*N;i++){A[i]=rand()/(double)RAND_MAX-0.5;B[i]=rand()/(double)RAND_MAX-0.5;}
 long M=4000000;double*H1=(double*)malloc(M*8),*H2=(double*)malloc(M*8);for(long i=0;i<M;i++){H1[i]=rand()/(double)RAND_MAX-0.5;H2[i]=rand()/(double)RAND_MAX-0.5;}
 printf("check %ld %ld %.9f %ld %.9f\n",fib(32),mandelbrot(1000,50),spectral(1000),sieve(10000000),nbody(1000000));
 T("fib(32)",sinkl=fib(V32)) T("mandelbrot 1000x1000 x50",sinkl=mandelbrot(V1000,V50)) T("spectral-norm 1000",sinkd=spectral(V1000))
 T("sieve 10M",sinkl=sieve(V10M)) T("nbody 1M steps",sinkd=nbody(V1M)) T("matmul 512 (naive ikj)",matmul(N,A,B,C))
 T("fused sqrt(a*a+b*b) sum 4M",sinkd=hypot_sum(M,H1,H2)) return 0;}
