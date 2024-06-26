/*
 * transform fft_serial to parallel with `myfor.h`
 */
# include "myfor.h"
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <float.h>
# include <omp.h>

#define NUM_THREADS 1
double ctime0;
double ctime1;
double ctime2;
double error;
int first;
double flops;
double fnm1;
int i;
int icase;
int it;
int ln2;
double mflops;
int n;
int nits = 10000;
static double seed;
double sgn;
double *w;
double *x;
double *y;
double *z;
double z0;
double z1;
const double pi = 3.141592653589793;

int main ( );
void ccopy ( int n, double x[], double y[] );
void cfft2 ( int n, double x[], double y[], double w[], double sgn );
void cffti ( int n, double w[] );
double cpu_time ( void );
double ggl ( double *ds );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
  double w[], double sgn );
void timestamp ( );

typedef struct{
  int mj; 
  double *a; 
  double *b; 
  double *c;
  double *d;
  double *w;
  double sgn; 
} temp_parallel ; 

temp_parallel temp; 

void* initialize_parallel(void* args) {
  ParallelType* p = (ParallelType*)args;
  int start = p->start;
  int end = p->end;
  int increment = p->increment;
  for (int index = start; index < end; index += increment) {
    x[index] = 0.0;
    z[index] = 0.0;           /* copy of initial real data */
    x[index + 1] = 0.0;
    z[index + 1] = 0.0;         /* copy of initial imag. data */
  }
  return NULL; 
}

void* cffti_parallel(void* args) {
  ParallelType *p = (ParallelType*)args;
  double aw = 2.0 * pi / ((double)n); 
  int start = p->start;
  int end = p->end;
  int increment = p->increment;
  for (int index = start; index < end; index += increment) {
    double arg = aw * ((double)index);
    w[index * 2 + 0] = cos(arg);
    w[index * 2 + 1] = sin(arg);
  }
  return NULL; 
}

void* step_parallel(void* args) {
  ParallelType *p = (ParallelType*)args;
  int start = p->start;
  int end = p->end;
  int increment = p->increment;
  temp_parallel arg = temp; 
  int mj = arg.mj; 
  double *a = arg.a; 
  double *b = arg.b;  
  double *c = arg.c; 
  double *d = arg.d; 
  double *w = arg.w; 
  double sgn = arg.sgn;

  for (int index = start; index < end; index += increment) {
    int jw = index * mj; 
    int ja = jw; 
    int jb = ja; 
    int jc = index * mj * 2; 
    int jd = jc; 
    double wjw[2] = {w[jw * 2 + 0], w[jw * 2 + 1]}; 
    if (sgn < 0.0) {
      wjw[1] = -wjw[1];
    }
    for (int index2 = 0; index2 < mj; index2++) {
      c[(jc + index2) * 2 + 0] = a[(ja + index2) * 2 + 0] + b[(jb + index2) * 2 + 0]; 
      c[(jc + index2) * 2 + 1] = a[(ja + index2) * 2 + 1] + b[(jb + index2) * 2 + 1]; 
      double ambr = a[(ja + index2) * 2 + 0] - b[(jb + index2) * 2 + 0]; 
      double ambu = a[(ja + index2) * 2 + 1] - b[(jb + index2) * 2 + 1]; 
      d[(jd + index2) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu; 
      d[(jd + index2) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu; 
    }
  }
}

/******************************************************************************/

int main ( ) {
  timestamp ( );
  printf ( "\n" );
  printf ( "FFT_SERIAL\n" );
  printf ( "  C version\n" );
  printf ( "\n" );
  printf ( "  Demonstrate an implementation of the Fast Fourier Transform\n" );
  printf ( "  of a complex data vector.\n" );
/*
  Prepare for tests.
*/
  printf ( "\n" );
  printf ( "  Accuracy check:\n" );
  printf ( "\n" );
  printf ( "    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n" );
  printf ( "\n" );
  printf ( "             N      NITS    Error         Time          Time/Call     MFLOPS\n" );
  printf ( "\n" );

  seed  = 331.0;
  n = 1;
/*
  LN2 is the log base 2 of N.  Each increase of LN2 doubles N.
*/
  for ( ln2 = 1; ln2 <= 20; ln2++ )
  {
    n = 2 * n;
/*
  Allocate storage for the complex arrays W, X, Y, Z.  

  We handle the complex arithmetic,
  and store a complex number as a pair of doubles, a complex vector as a doubly
  dimensioned array whose second dimension is 2. 
*/
    w = ( double * ) malloc (     n * sizeof ( double ) );
    x = ( double * ) malloc ( 2 * n * sizeof ( double ) );
    y = ( double * ) malloc ( 2 * n * sizeof ( double ) );
    z = ( double * ) malloc ( 2 * n * sizeof ( double ) );

    first = 1;

    
    for ( icase = 0; icase < 2; icase++ )
    {
      if ( first )
      {
        for ( i = 0; i < 2 * n; i = i + 2 )
        {
          z0 = ggl ( &seed );
          z1 = ggl ( &seed );
          x[i] = z0;
          z[i] = z0;
          x[i+1] = z1;
          z[i+1] = z1;
        }
      } 
      else // need to parallel
      {
        parallel_for(0, 2 * n, 2, initialize_parallel, NULL, NUM_THREADS);
      }
/* 
  Initialize the sine and cosine tables.
*/
      cffti ( n, w );
/* 
  Transform forward, back 
*/
      if ( first )
      {
        sgn = + 1.0;
        cfft2 ( n, x, y, w, sgn );
        sgn = - 1.0;
        cfft2 ( n, y, x, w, sgn );
/* 
  Results should be same as the initial data multiplied by N.
*/
        fnm1 = 1.0 / ( double ) n;
        error = 0.0;
        for ( i = 0; i < 2 * n; i = i + 2 )
        {
          error = error 
          + pow ( z[i]   - fnm1 * x[i], 2 )
          + pow ( z[i+1] - fnm1 * x[i+1], 2 );
        }
        error = sqrt ( fnm1 * error );
        printf ( "  %12d  %8d  %12e", n, nits, error );
        first = 0;
      }
      else
      {
        ctime1 = cpu_time ( );
        for ( it = 0; it < nits; it++ )
        {
          sgn = + 1.0;
          cfft2 ( n, x, y, w, sgn );
          sgn = - 1.0;
          cfft2 ( n, y, x, w, sgn );
        }
        ctime2 = cpu_time ( );
        ctime0 = ctime2 - ctime1;

        flops = 2.0 * ( double ) nits * ( 5.0 * ( double ) n * ( double ) ln2 );

        mflops = flops / 1.0E+06 / ctime0;

        printf ( "  %12e  %12e  %12f\n", ctime0, ctime0 / ( double ) ( 2 * nits ), mflops );
      }
    }
    if ( ( ln2 % 4 ) == 0 ) 
    {
      nits = nits / 10;
    }
    if ( nits < 1 ) 
    {
      nits = 1;
    }
    free ( w );
    free ( x );
    free ( y );
    free ( z );
  }
  printf ( "\n" );
  printf ( "FFT_SERIAL:\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}
/******************************************************************************/

void ccopy ( int n, double x[], double y[] )

/******************************************************************************/
/*
  Purpose:

    CCOPY copies a complex vector.

  Discussion:

    The "complex" vector A[N] is actually stored as a double vector B[2*N].

    The "complex" vector entry A[I] is stored as:

      B[I*2+0], the real part,
      B[I*2+1], the imaginary part.

  Modified:

    23 March 2009

  Author:

    Original C version by Wesley Petersen.
    This C version by John Burkardt.

  Reference:

    Wesley Petersen, Peter Arbenz, 
    Introduction to Parallel Computing - A practical guide with examples in C,
    Oxford University Press,
    ISBN: 0-19-851576-6,
    LC: QA76.58.P47.

  Parameters:

    Input, int N, the length of the vector.

    Input, double X[2*N], the vector to be copied.

    Output, double Y[2*N], a copy of X.
*/
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    y[i*2+0] = x[i*2+0];
    y[i*2+1] = x[i*2+1];
   }
  return;
}
/******************************************************************************/

void cfft2 ( int n, double x[], double y[], double w[], double sgn )

/******************************************************************************/
/*
  Purpose:

    CFFT2 performs a complex Fast Fourier Transform.

  Modified:

    23 March 2009

  Author:

    Original C version by Wesley Petersen.
    This C version by John Burkardt.

  Reference:

    Wesley Petersen, Peter Arbenz, 
    Introduction to Parallel Computing - A practical guide with examples in C,
    Oxford University Press,
    ISBN: 0-19-851576-6,
    LC: QA76.58.P47.

  Parameters:

    Input, int N, the size of the array to be transformed.

    Input/output, double X[2*N], the data to be transformed.  
    On output, the contents of X have been overwritten by work information.

    Output, double Y[2*N], the forward or backward FFT of X.

    Input, double W[N], a table of sines and cosines.

    Input, double SGN, is +1 for a "forward" FFT and -1 for a "backward" FFT.
*/
{
  int j;
  int m;
  int mj;
  int tgle;

   m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
   mj   = 1;
/*
  Toggling switch for work array.
*/
  tgle = 1;
  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

  if ( n == 2 )
  {
    return;
  }

  for ( j = 0; j < m - 2; j++ )
  {
    mj = mj * 2;
    if ( tgle )
    {
      step ( n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn );
      tgle = 0;
    }
    else
    {
      step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
      tgle = 1;
    }
  }
/* 
  Last pass through data: move Y to X if needed.
*/
  if ( tgle ) 
  {
    ccopy ( n, y, x );
  }

  mj = n / 2;
  step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

  return;
}
/******************************************************************************/

void cffti ( int n, double w[] ) {
  double arg;
  double aw;
  int i;
  int n2;
  const double pi = 3.141592653589793;

  n2 = n / 2;
  aw = 2.0 * pi / ( ( double ) n );
  
  // need to parallel
  parallel_for(0, n2, 1, cffti_parallel, NULL, NUM_THREADS); 
  return;
}
/*******************************************************************************/

double cpu_time ( void )

/*******************************************************************************/
/*
  Purpose:
 
    CPU_TIME reports the total CPU time for a program.

  Modified:

    27 September 2005

  Author:

    John Burkardt

  Parameters:

    Output, double CPU_TIME, the current total elapsed CPU time in second.
*/
{
  double value;

  value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;

  return value;
}
/******************************************************************************/

double ggl ( double *seed )

/******************************************************************************/
/* 
  Purpose:

    GGL generates uniformly distributed pseudorandom real numbers in [0,1]. 

  Modified:

    23 March 2009

  Author:

    Original C version by Wesley Petersen, M Troyer, I Vattulainen.
    This C version by John Burkardt.

  Reference:

    Wesley Petersen, Peter Arbenz, 
    Introduction to Parallel Computing - A practical guide with examples in C,
    Oxford University Press,
    ISBN: 0-19-851576-6,
    LC: QA76.58.P47.

  Parameters:

    Input/output, double *SEED, used as a seed for the sequence.

    Output, double GGL, the next pseudorandom value.
*/
{
  double d2 = 0.2147483647e10;
  double t;
  double value;

  t = *seed;
  t = fmod ( 16807.0 * t, d2 );
  *seed = t;
  value = ( t - 1.0 ) / ( d2 - 1.0 );

  return value;
}
/******************************************************************************/

void step ( int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn ) {
    temp.mj = mj; 
    temp.a = a; 
    temp.b = b; 
    temp.c = c; 
    temp.d = d; 
    temp.w = w; 
    temp.sgn = sgn; 

    int mj2 = 2 * mj;
    int lj  = n / mj2;

    parallel_for(0, lj, 1, step_parallel, NULL, NUM_THREADS); 
}
/******************************************************************************/

void timestamp ( )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}

