//
// Created by martin on 26/10/21.
//
#include <jni.h>
#include <android/bitmap.h>
#include <stdlib.h>
#include "x86_64/halide_dens_step.h"
#include "x86_64/halide_vel_step.h"
#include "x86_64/halide_bitmap.h"


#include "HalideBuffer.h"

#include "HalideRuntimeOpenGLCompute.h"
#include "HalideRuntimeOpenCL.h"

using namespace Halide::Runtime;

#define IX(i, j) ((j)+(NW+2)*(i))
#define SWAP(x0, x) {float * tmp=x0;x0=x;x=tmp;}
#define FOR_EACH_CELL for ( i=1 ; i<=NH ; i++ ) { for ( j=1 ; j<=NW ; j++ ) {
#define END_FOR }}


void add_source(int NW, int NH, float *x, float *s, float dt) {
    int i, size = (NW + 2) * (NH + 2);
    for (i = 0; i < size; i++)
        x[i] += dt * s[i];
}

void lin_solve(int NW, int NH, float *x, float *x0, float a, float c, int iterations) {
    int i, j, k;
    for (k = 0; k < iterations; k++) {
        FOR_EACH_CELL
                x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] +
                        x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / c;
        END_FOR
    }
}
void diffuse(int NW, int NH, float *x, float *x0, float diff, float dt) {
    float a = dt * diff * NW * NH;
    lin_solve(NW, NH, x, x0, a, 1 + 4 * a, 2);
}

void advect(int NW, int NH, float *d, float *d0, float *u, float *v, float dt) {
    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;

    dt0 = dt * NH;
    FOR_EACH_CELL
            x = i - dt0 * u[IX(i, j)];
            y = j - dt0 * v[IX(i, j)];
            if (x < 0.5f) x = 0.5f;
            if (x > NH + 0.5f) x = NH + 0.5f;
            i0 = (int) x;
            i1 = i0 + 1;
            if (y < 0.5f) y = 0.5f;
            if (y > NW + 0.5f) y = NW + 0.5f;
            j0 = (int) y;
            j1 = j0 + 1;
            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;
            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                          s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    END_FOR
}

void project(int NW, int NH, float *u, float *v, float *p, float *div) {
    int i, j;

    FOR_EACH_CELL
            div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] -
                                     v[IX(i, j - 1)]) / NH;
            p[IX(i, j)] = 0;
    END_FOR

    lin_solve(NW, NH, p, div, 1, 4, 10);

    FOR_EACH_CELL
            u[IX(i, j)] -= 0.5f * NH * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            v[IX(i, j)] -= 0.5f * NH * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
    END_FOR
}

void dens_step(int NW, int NH, float *x, float *x0, float *u, float *v, float diff, float dt) {
    add_source(NW, NH, x, x0, dt);
    diffuse(NW, NH, x0, x, diff, dt);
    advect(NW, NH, x, x0, u, v, dt);
}

void vel_step(int NW, int NH, float *u, float *v, float *u0, float *v0, float visc, float dt) {
    add_source(NW, NH, u, u0, dt);
    add_source(NW, NH, v, v0, dt);
    diffuse(NW, NH, u0, u, visc, dt);
    diffuse(NW, NH, v0, v, visc, dt);
    project(NW, NH, u0, v0, u, v);
    advect(NW, NH, u, u0, u0, v0, dt);
    advect(NW, NH, v, v0, u0, v0, dt);
    project(NW, NH, u, v, u0, v0);
}
static int width, height;
static float dt, diff, visc;
static int halide=0;
static float *u = 0, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;
Buffer<float> u_h, v_h, u0_h, v0_h, dens_h, dens0_h;
extern "C" {
JNIEXPORT void JNICALL
Java_com_example_martin_simulation_NativeSimulation_init(JNIEnv *env, jobject thiz, jint w,
                                                           jint h, jint usehalide) {
    width = w;
    height = h;
    halide = usehalide;
    int sz=(w+2)*(h+2)*4;
    if (u == 0) {
        int h2 = h + 2;
        int w2 = w + 2;
        u_h = Buffer<float>(w2, h2);
        u0_h = Buffer<float>(w2, h2);
        v_h = Buffer<float>(w2, h2);
        v0_h = Buffer<float>(w2, h2);
        dens_h = Buffer<float>(w2, h2);
        dens0_h = Buffer<float>(w2, h2);
        u = (float *) (*u_h).host;
        v = (float *) (*v_h).host;
        u_prev = (float *) (*u0_h).host;
        v_prev = (float *) (*v0_h).host;
        dens = (float *) (*dens_h).host;
        dens_prev = (float *) (*dens0_h).host;
    }
    memset(u,0,sz);
    memset(v,0,sz);
    memset(dens,0,sz);
    memset(u_prev,0,sz);
    memset(v_prev,0,sz);
    memset(dens_prev,0,sz);
    u_h.set_host_dirty();
    v_h.set_host_dirty();
}

JNIEXPORT void JNICALL
Java_com_example_martin_simulation_NativeSimulation_ui_1update(JNIEnv *env, jobject thiz,
                                                                 jfloat x, jfloat y, jfloat px,
                                                                 jfloat py, jfloat source, jfloat force, int flames) {
    int NW = width;
    int sz=(height+2)*(width+2)*4;
    memset(dens_prev,0,sz);
    memset(v_prev,0,sz);
    memset(u_prev,0,sz);
    for(int i=0;i<flames;i++) {
        int xp = width / (flames + 1) * (i + 1);
        dens_prev[IX(height - 10, xp)] = source;
        v_prev[IX(height - 10, xp)] = 0;
        u_prev[IX(height - 10, xp)] = -force;
    }
    if(x<0) return;
    for(int xx=(int) x-1; xx<(int)x+2; xx++)
        for(int yy=(int) y-1; yy<(int)y+2; yy++) {
            if (xx > 0 && yy > 0 && xx < width && yy < height) {
                dens_prev[IX(yy, xx)] = source;
                if (px > 0) {
                    v_prev[IX(yy, xx)] = force * (x - px);
                    u_prev[IX(yy, xx)] = force * (y - py);
                }
            }
        }

}
JNIEXPORT void JNICALL
Java_com_example_martin_simulation_NativeSimulation_dens_1step(JNIEnv *env, jobject thiz, jfloat diff, jfloat dt) {
    if(halide>1) {
        dens0_h.set_host_dirty();
        halide_dens_step(dens_h, dens0_h, u_h, v_h, diff, dt, dens_h);
    }
    else
        dens_step(width, height, dens, dens_prev, u, v, diff, dt);
}
JNIEXPORT void JNICALL
Java_com_example_martin_simulation_NativeSimulation_vel_1step(JNIEnv *env, jobject thiz, jfloat visc, jfloat dt) {
   if(halide>1) {
       u0_h.set_host_dirty();
       v0_h.set_host_dirty();
       halide_vel_step(u_h, v_h, u0_h, v0_h, visc, dt, u_h, v_h);
   }
   else
       vel_step(width, height, u, v, u_prev, v_prev, visc, dt);
}
JNIEXPORT void JNICALL
Java_com_example_martin_simulation_NativeSimulation_fillBitmap(JNIEnv *env, jobject thiz, jobject bitmap) {
    int NW = width;
    unsigned *src;
    int rc=AndroidBitmap_lockPixels(env, bitmap, (void **) &src);
    if(rc!=ANDROID_BITMAP_RESULT_SUCCESS)
        return;
    if(halide>1) {
        Buffer<unsigned > bmp(src,width,height);
        halide_bitmap(dens_h,u_h,v_h,bmp);
        bmp.copy_to_host();
    } else {
        for (int i = 1; i <= height; i++) {
            for (int j = 1; j <= width; j++) {
                int r = (int) (dens[IX(i, j)] * 255);
                int g = (int) (u[IX(i, j)] * -2000) + 128;
                int b = (int) (v[IX(i, j)] * 2000) + 128;
                if (r > 255) r = 255;
                if (r < 0) r = 0;
                if (g > 255) g = 255;
                if (g < 0) g = 0;
                if (b > 255) b = 255;
                if (b < 0) b = 0;
                int c = r | (g << 8) | (b << 16) | 0xff000000;
                src[(i - 1) * width + j - 1] = c;
            }
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
}
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_martin_simulation_NativeSimulation_shutdown(JNIEnv *env, jobject thiz) {
    auto dev = u_h.raw_buffer()->device_interface;
    u_h.deallocate();
    u0_h.deallocate();
    v_h.deallocate();
    v0_h.deallocate();
    dens_h.deallocate();
    dens0_h.deallocate();
    u=0;
    if (dev) {
        halide_device_release(nullptr, dev);
    }
    halide_profiler_report(nullptr);
}