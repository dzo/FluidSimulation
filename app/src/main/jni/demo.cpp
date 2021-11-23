/*
  ======================================================================
   demo.c --- protoype to show off the simple solver
  ----------------------------------------------------------------------
   Author : Jos Stam (jstam@aw.sgi.com)
   Creation Date : Jan 9 2003

   Description:

	This code is a simple prototype that demonstrates how to use the
	code provided in my GDC2003 paper entitles "Real-Time Fluid Dynamics
	for Games". This code uses OpenGL and GLUT for graphics and interface

  =======================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <GL/glut.h>
#include <time.h>
#include  <signal.h>

#include "host/halide_dens_step.h"
#include "host/halide_vel_step.h"
#include "HalideBuffer.h"

//#include "HalideRuntimeOpenGLCompute.h"

using namespace Halide::Runtime;

/* macros */

#define IX(i,j) ((i)+(N+2)*(j))

/* external definitions (from solver.c) */

extern void dens_step ( int N, float * x, float * x0, float * u, float * v, float diff, float dt );
extern void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt );

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;
static int flames=3;
static int dvel;

static float * u, * v, * u_prev, * v_prev;
static float * dens, * dens_prev;

Buffer<float> u_h, v_h, u0_h, v0_h, dens_h, dens0_h;

static int win_id;
static int win_x, win_y;
static int mouse_down[3];
static int omx, omy, mx, my;


/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/


static void free_data ( void )
{
	u_h.deallocate();
	v_h.deallocate();
	u0_h.deallocate();
	v0_h.deallocate();
	dens_h.deallocate();
	dens0_h.deallocate();
}

static void clear_data ( void )
{
	int i, size=(N+2)*(N+2);

	for ( i=0 ; i<size ; i++ ) {
		u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
	}
}

static int allocate_data ( void )
{
	int h2=N+2;
	int w2=h2;
	int size = (N+2)*(N+2);
	u_h=Buffer<float>(w2,h2);
	u0_h=Buffer<float>(w2,h2);
	v_h=Buffer<float>(w2,h2);
	v0_h=Buffer<float>(w2,h2);
	dens_h=Buffer<float>(w2,h2);
	dens0_h=Buffer<float>(w2,h2);
	u			= (float *)(*u_h).host;
	v			= (float *)(*v_h).host;
	u_prev		= (float *)(*u0_h).host;
	v_prev		= (float *)(*v0_h).host;
	dens		= (float *)(*dens_h).host;
	dens_prev	= (float *)(*dens0_h).host;

	if ( !u || !v || !u_prev || !v_prev || !dens || !dens_prev ) {
		fprintf ( stderr, "cannot allocate data\n" );
		return ( 0 );
	}

	return ( 1 );
}


/*
  ----------------------------------------------------------------------
   OpenGL specific drawing routines
  ----------------------------------------------------------------------
*/

static void pre_display ( void )
{
	glViewport ( 0, 0, win_x, win_y );
	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity ();
	gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT );
}

static void post_display ( void )
{
	glutSwapBuffers ();
}

static void draw_velocity ( void )
{
	int i, j;
	float x, y, h;

	h = 1.0f/N;

	glColor3f ( 1.0f, 1.0f, 1.0f );
	glLineWidth ( 1.0f );

	glBegin ( GL_LINES );

		for ( i=1 ; i<=N ; i++ ) {
			x = (i-0.5f)*h;
			for ( j=1 ; j<=N ; j++ ) {
				y = (j-0.5f)*h;
				glVertex2f ( x, y );
				glVertex2f ( x+u[IX(i,j)], y+v[IX(i,j)] );
			}
		}

	glEnd ();
}

unsigned *src=0;

void fillbitmap() {
    if (src==0) src=(unsigned *)malloc((N+2)*(N+2)*4);
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            int r = (int) (dens[IX(i, j)] * 255);
            int g = (int) (u[IX(i, j)] * 2550) + 128;
            int b = (int) (v[IX(i, j)] * 2550) + 128;
            if (r > 255) r = 255;
            if (r < 0) r = 0;
            if (g > 255) g = 255;
            if (g < 0) g = 0;
            if (b > 255) b = 255;
            if (b < 0) b = 0;
            int c = r | (g << 8) | (b << 16) | 0xff000000;
            src[(j - 1) * N + i - 1] = c;
        }
    }
}
static void draw_density ( void )
{
    fillbitmap();
    glPixelZoom((float)win_x/N,(float)win_y/N);
    glDrawPixels(N,N,GL_RGBA,GL_UNSIGNED_BYTE,src);
    /*
	int i, j;
	float x, y, h, d00, d01, d10, d11,r,g,b;

	h = 1.0f/N;

	glBegin ( GL_QUADS );

		for ( i=0 ; i<=N ; i++ ) {
			x = (i-0.5f)*h;
			for ( j=0 ; j<=N ; j++ ) {
				y = (j-0.5f)*h;

				r = dens[IX(i,j)];
				g = 100*(v[IX(i,j)])+0.5;
				b = 100*(u[IX(i,j)])+0.5;

				glColor3f ( r, g, b );
				glVertex2f ( x, y );
				glVertex2f ( x+h, y );
				glVertex2f ( x+h, y+h );
				glVertex2f ( x, y+h );
			}
		}

	glEnd ();
     */
}

/*
  ----------------------------------------------------------------------
   relates mouse movements to forces sources
  ----------------------------------------------------------------------
*/

static void get_from_UI ( float * d, float * u, float * v )
{
	int i, j, size = (N+2)*(N+2);

	for ( i=0 ; i<size ; i++ ) {
		u[i] = v[i] = d[i] = 0.0f;
	}

	for(int i=0;i<flames;i++) {
		int xp = N / (flames + 1) * (i + 1);
		d[IX(xp, 10)] = source;
		v[IX(xp, 10)] = 0;
		u[IX(xp, 10)] = force;
	}
	if ( !mouse_down[0] && !mouse_down[2] ) return;

	i = (int)((       mx /(float)win_x)*N+1);
	j = (int)(((win_y-my)/(float)win_y)*N+1);

	if ( i<1 || i>N || j<1 || j>N ) return;


	if ( mouse_down[0] ) {
	    for(int x=i-1;x<i+2;x++)
	        for(int y=j-1;y<j+2;y++)
	            if (x>1 && x<N && y>1 && y<N) {
                    u[IX(x, y)] = force * (mx - omx);
                    v[IX(x, y)] = force * (omy - my);
                }
	}

	if ( mouse_down[2] ) {
        for(int x=i-1;x<i+2;x++)
            for(int y=j-1;y<j+2;y++)
                if (x>1 && x<N && y>1 && y<N) {
                    d[IX(x, y)] = source;
                }
	}

	omx = mx;
	omy = my;

	return;
}

/*
  ----------------------------------------------------------------------
   GLUT callback routines
  ----------------------------------------------------------------------
*/

static void key_func ( unsigned char key, int x, int y )
{
	switch ( key )
	{
		case 'c':
		case 'C':
			clear_data ();
			break;

		case 'q':
		case 'Q':
			free_data ();
//			halide_device_release(nullptr, halide_cuda_device_interface());
			exit ( 0 );
			break;

		case 'v':
		case 'V':
			dvel = !dvel;
			break;
	}
}

static void mouse_func ( int button, int state, int x, int y )
{
	omx = mx = x;
	omx = my = y;

	mouse_down[button] = state == GLUT_DOWN;
}

static void motion_func ( int x, int y )
{
	mx = x;
	my = y;
}

static void reshape_func ( int width, int height )
{
	glutSetWindow ( win_id );
	glutReshapeWindow ( width, height );

	win_x = width;
	win_y = height;
}

double lasttime;

double get_time() {
	struct timespec time;
	clock_gettime(CLOCK_REALTIME,&time);
	return time.tv_sec+time.tv_nsec/1e9;
}
static int frames=0;
static void idle_func ( void )
{
	get_from_UI ( dens_prev, u_prev, v_prev );
	double time,time1,time2,newtime;
 	time=get_time();
#if NOHALIDE
	time1=get_time();
	vel_step ( N, u, v, u_prev, v_prev, visc, dt );
	time2=get_time();
	dens_step ( N, dens, dens_prev, u, v, diff, dt );
#else
 	u0_h.set_host_dirty();
	v0_h.set_host_dirty();
    	dens0_h.set_host_dirty();
    	time1=get_time();
	halide_vel_step(u_h, v_h, u0_h, v0_h, visc, dt, u_h, v_h);
	halide_dens_step(dens_h,dens0_h,u_h,v_h,diff,dt, dens_h);
	time2=get_time();
    	u_h.copy_to_host();
    	v_h.copy_to_host();
	dens_h.copy_to_host();
//	dens_h.device_sync();
//	u_h.device_sync();
//    v_h.device_sync();
#endif
	frames++;
	if(time-lasttime>1.0) {
		lasttime=time;
		newtime=get_time();
		printf("Fps:%f %f %d\n", 1.0 / (newtime-time), 1.0/(time2-time1),frames);
		frames=0;
	}
#ifndef NOUI
	glutSetWindow ( win_id );
	glutPostRedisplay ();
#endif
}

static void display_func ( void )
{
	pre_display ();
	 u_h.copy_to_host();
        v_h.copy_to_host();
        dens_h.copy_to_host();

		if ( dvel ) draw_velocity ();
		else		draw_density ();

	post_display ();
}


/*
  ----------------------------------------------------------------------
   open_glut_window --- open a glut compatible window and set callbacks
  ----------------------------------------------------------------------
*/

static void open_glut_window ( void )
{
	glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE );

	glutInitWindowPosition ( 0, 0 );
	glutInitWindowSize ( win_x, win_y );
	win_id = glutCreateWindow ( "Navier Stokes" );

	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT );
	glutSwapBuffers ();
	glClear ( GL_COLOR_BUFFER_BIT );
	glutSwapBuffers ();

	pre_display ();

	glutKeyboardFunc ( key_func );
	glutMouseFunc ( mouse_func );
	glutMotionFunc ( motion_func );
	glutReshapeFunc ( reshape_func );
	glutIdleFunc ( idle_func );
	glutDisplayFunc ( display_func );
}


/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/
void  INThandler(int sig) {
	exit(0);
}

int main ( int argc, char ** argv )
{
	glutInit ( &argc, argv );

	if ( argc != 1 && argc != 7 ) {
		fprintf ( stderr, "usage : %s N dt diff visc force source\n", argv[0] );
		fprintf ( stderr, "where:\n" );\
		fprintf ( stderr, "\t N      : grid resolution\n" );
		fprintf ( stderr, "\t dt     : time step\n" );
		fprintf ( stderr, "\t diff   : diffusion rate of the density\n" );
		fprintf ( stderr, "\t visc   : viscosity of the fluid\n" );
		fprintf ( stderr, "\t force  : scales the mouse movement that generate a force\n" );
		fprintf ( stderr, "\t source : amount of density that will be deposited\n" );
//		exit ( 1 );
	}

	if ( argc == 1 ) {
		N = 64;
		dt = 0.1f;
		diff = 0.0f;
		visc = 0.0f;
		force = 5.0f;
		source = 100.0f;
		fprintf ( stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
			N, dt, diff, visc, force, source );
	} else {
		N = atoi(argv[1]);
		dt = atof(argv[2]);
		diff = atof(argv[3]);
		visc = atof(argv[4]);
		force = atof(argv[5]);
		source = atof(argv[6]);
	}

	printf ( "\n\nHow to use this demo:\n\n" );
	printf ( "\t Add densities with the right mouse button\n" );
	printf ( "\t Add velocities with the left mouse button and dragging the mouse\n" );
	printf ( "\t Toggle density/velocity display with the 'v' key\n" );
	printf ( "\t Clear the simulation by pressing the 'c' key\n" );
	printf ( "\t Quit by pressing the 'q' key\n" );

	dvel = 0;

	if ( !allocate_data () ) exit ( 1 );
	clear_data ();

	signal(SIGINT, INThandler);
#ifdef NOUI
	while(1) idle_func();
#endif

	win_x = 512;
	win_y = 512;
	open_glut_window ();

	glutMainLoop ();

	exit ( 0 );
}
