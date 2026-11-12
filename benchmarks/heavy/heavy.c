/* Heavy CPU benchmarks in C — same algorithms, sizes and random numbers as heavy.apsi. Build: gcc -O3 -march=native heavy.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static inline long rng_next(long s){return (s*16807)%2147483647;}
#define U(s) ((double)(s)/2147483647.0)

double nbody_dense(long n,long steps){
 double *x=calloc(n,8),*y=calloc(n,8),*z=calloc(n,8),*vx=calloc(n,8),*vy=calloc(n,8),*vz=calloc(n,8),
        *ax=calloc(n,8),*ay=calloc(n,8),*az=calloc(n,8),*m=calloc(n,8);
 long s=42;
 for(long i=0;i<n;i++){s=rng_next(s);x[i]=U(s)-0.5;s=rng_next(s);y[i]=U(s)-0.5;s=rng_next(s);z[i]=U(s)-0.5;
  s=rng_next(s);m[i]=1.0/n*(0.5+U(s));}
 double dt=0.001,eps2=0.0001;
 for(long st=0;st<steps;st++){
  for(long i=0;i<n;i++){double fx=0,fy=0,fz=0,xi=x[i],yi=y[i],zi=z[i];
   for(long j=0;j<n;j++){double dx=x[j]-xi,dy=y[j]-yi,dz=z[j]-zi;double d2=dx*dx+dy*dy+dz*dz+eps2;
    double inv=m[j]/(d2*sqrt(d2));fx+=dx*inv;fy+=dy*inv;fz+=dz*inv;}
   ax[i]=fx;ay[i]=fy;az[i]=fz;}
  for(long i=0;i<n;i++){vx[i]+=dt*ax[i];vy[i]+=dt*ay[i];vz[i]+=dt*az[i];x[i]+=dt*vx[i];y[i]+=dt*vy[i];z[i]+=dt*vz[i];}}
 double e=0;for(long i=0;i<n;i++)e+=0.5*m[i]*(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]);
 free(x);free(y);free(z);free(vx);free(vy);free(vz);free(ax);free(ay);free(az);free(m);return e;}

double jacobi(long n,long iters){
 double *a=calloc(n*n,8),*b=calloc(n*n,8);
 for(long j=0;j<n;j++){a[j]=100.0;b[j]=100.0;}
 for(long it=0;it<iters;it++){
  for(long i=1;i<n-1;i++)for(long j=1;j<n-1;j++)b[i*n+j]=0.25*(a[(i-1)*n+j]+a[(i+1)*n+j]+a[i*n+j-1]+a[i*n+j+1]);
  for(long i=1;i<n-1;i++)for(long j=1;j<n-1;j++)a[i*n+j]=0.25*(b[(i-1)*n+j]+b[(i+1)*n+j]+b[i*n+j-1]+b[i*n+j+1]);}
 double s=0;for(long i=0;i<n*n;i++)s+=a[i];free(a);free(b);return s;}

double matmul(long n){
 double *a=calloc(n*n,8),*b=calloc(n*n,8),*c=calloc(n*n,8);long s=7;
 for(long i=0;i<n;i++)for(long j=0;j<n;j++){s=rng_next(s);a[i*n+j]=U(s)-0.5;s=rng_next(s);b[i*n+j]=U(s)-0.5;}
 for(long i=0;i<n;i++)for(long k=0;k<n;k++){double aik=a[i*n+k];for(long j=0;j<n;j++)c[i*n+j]+=aik*b[k*n+j];}
 double t=0;for(long i=0;i<n;i++)t+=c[i*n+(i*7)%n];free(a);free(b);free(c);return t;}

double lu_logdet(long n){
 double *a=calloc(n*n,8);long s=11;
 for(long i=0;i<n;i++)for(long j=0;j<n;j++){s=rng_next(s);a[i*n+j]=U(s)-0.5;}
 double logdet=0;
 for(long k=0;k<n;k++){long p=k;double best=fabs(a[k*n+k]);
  for(long i=k+1;i<n;i++){double v=fabs(a[i*n+k]);if(v>best){best=v;p=i;}}
  if(p!=k)for(long j=0;j<n;j++){double t=a[k*n+j];a[k*n+j]=a[p*n+j];a[p*n+j]=t;}
  double piv=a[k*n+k];logdet+=log(fabs(piv));
  for(long i=k+1;i<n;i++){double f=a[i*n+k]/piv;a[i*n+k]=f;for(long j=k+1;j<n;j++)a[i*n+j]-=f*a[k*n+j];}}
 free(a);return logdet;}

long ising(long n,long sweeps,double beta){
 double *sp=calloc(n*n,8);long s=5;
 for(long i=0;i<n;i++)for(long j=0;j<n;j++){s=rng_next(s);sp[i*n+j]=s<1073741824?1.0:-1.0;}
 double w4=exp(-2.0*beta*4.0),w8=exp(-2.0*beta*8.0);
 for(long sw=0;sw<sweeps;sw++)for(long i=0;i<n;i++)for(long j=0;j<n;j++){
  double nb=sp[((i+1)%n)*n+j]+sp[((i+n-1)%n)*n+j]+sp[i*n+(j+1)%n]+sp[i*n+(j+n-1)%n];
  double de=2.0*sp[i*n+j]*nb;s=rng_next(s);double r=U(s);
  if(de<=0.0||(de==4.0&&r<w4)||(de==8.0&&r<w8))sp[i*n+j]=-sp[i*n+j];}
 long mag=0;for(long i=0;i<n*n;i++)mag+=(long)sp[i];free(sp);return mag;}

long levenshtein(long n){
 double *a=calloc(n,8),*b=calloc(n,8),*prev=calloc(n+1,8),*cur=calloc(n+1,8);long s=3;
 for(long i=0;i<n;i++){s=rng_next(s);a[i]=(double)(s%4);s=rng_next(s);b[i]=(double)(s%4);}
 for(long j=0;j<=n;j++)prev[j]=(double)j;
 for(long i=1;i<=n;i++){cur[0]=(double)i;double ai=a[i-1];
  for(long j=1;j<=n;j++){double cost=ai==b[j-1]?0.0:1.0;cur[j]=fmin(fmin(prev[j]+1.0,cur[j-1]+1.0),prev[j-1]+cost);}
  for(long j=0;j<=n;j++)prev[j]=cur[j];}
 long r=(long)prev[n];free(a);free(b);free(prev);free(cur);return r;}

double vdp_rk4(long steps){
 double x=2.0,v=0.0,h=0.0001,mu=2.0;
 for(long i=0;i<steps;i++){
  double k1x=v,k1v=mu*(1.0-x*x)*v-x;
  double x2=x+0.5*h*k1x,v2=v+0.5*h*k1v,k2x=v2,k2v=mu*(1.0-x2*x2)*v2-x2;
  double x3=x+0.5*h*k2x,v3=v+0.5*h*k2v,k3x=v3,k3v=mu*(1.0-x3*x3)*v3-x3;
  double x4=x+h*k3x,v4=v+h*k3v,k4x=v4,k4v=mu*(1.0-x4*x4)*v4-x4;
  x+=h/6.0*(k1x+2.0*k2x+2.0*k3x+k4x);v+=h/6.0*(k1v+2.0*k2v+2.0*k3v+k4v);}
 return x+v;}

double kmeans(long n,long k,long iters){
 double *px=calloc(n,8),*py=calloc(n,8),*cx=calloc(k,8),*cy=calloc(k,8),*sx=calloc(k,8),*sy=calloc(k,8),*cnt=calloc(k,8);long s=9;
 for(long i=0;i<n;i++){s=rng_next(s);long c=s%k;s=rng_next(s);px[i]=(double)(c%4)*3.0+U(s);s=rng_next(s);py[i]=(double)(c/4)*3.0+U(s);}
 for(long c=0;c<k;c++){cx[c]=px[c];cy[c]=py[c];}
 double inertia=0;
 for(long it=0;it<iters;it++){
  for(long c=0;c<k;c++){sx[c]=0;sy[c]=0;cnt[c]=0;}
  inertia=0;
  for(long i=0;i<n;i++){long bc=0;double bd=1e300;
   for(long c=0;c<k;c++){double dx=px[i]-cx[c],dy=py[i]-cy[c];double d=dx*dx+dy*dy;if(d<bd){bd=d;bc=c;}}
   sx[bc]+=px[i];sy[bc]+=py[i];cnt[bc]+=1.0;inertia+=bd;}
  for(long c=0;c<k;c++)if(cnt[c]>0){cx[c]=sx[c]/cnt[c];cy[c]=sy[c]/cnt[c];}}
 free(px);free(py);free(cx);free(cy);free(sx);free(sy);free(cnt);return inertia;}

volatile long N2000=2000,N10=10,N1024=1024,N200=200,N800=800,N512=512,N50=50,N10K=10000,N20M=20000000,N200K=200000,N16=16,N20=20;
#define T(name,type,fmt,expr) { type r=expr; double best=1e9; for(int k_=0;k_<3;k_++){double t=now(); volatile type q=expr; (void)q; double d=now()-t; if(d<best)best=d;} printf("%s: %.1f ms  [check " fmt "]\n",name,best*1000,r); }
int main(void){
 T("nbody-dense 2000x10",double,"%.12g",nbody_dense(N2000,N10))
 T("jacobi-2d 1024x200",double,"%.12g",jacobi(N1024,N200))
 T("matmul 1024",double,"%.12g",matmul(N1024))
 T("lu-pivot 800",double,"%.12g",lu_logdet(N800))
 T("ising 512x50",long,"%ld",ising(N512,N50,0.44))
 T("levenshtein 10k",long,"%ld",levenshtein(N10K))
 T("ode-rk4 20M",double,"%.12g",vdp_rk4(N20M))
 T("kmeans 200k k16 x20",double,"%.12g",kmeans(N200K,N16,N20))
 return 0;}
