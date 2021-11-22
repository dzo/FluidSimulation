package com.example.martin.simulation;

import android.graphics.Bitmap;

public class JavaSimulation implements Simulation {

    float [][] u,  v,  u0,  v0;
    float [][] x,  x0;
    int width,height;
    int w2,h2;

    JavaSimulation(int w, int h) {
        width=w;
        height=h;
        w2=w+2;
        h2=h+2;
        u=new float[h2][w2];
        v=new float[h2][w2];
        u0=new float[h2][w2];
        v0=new float[h2][w2];
        x=new float[h2][w2];
        x0=new float[h2][w2];
    }

    void add_source( float[][] x, float[][] s, float dt) {
        for (int i = 0; i < h2; i++)
            for (int j = 0; j < w2; j++)
                x[i][j] += dt * s[i][j];
    }

    void lin_solve( float[][] x, float[][] x0, float a, float c, int iters) {
        for (int k = 0; k < iters; k++) {
            for (int i = 1; i <= height; i++) {
                for (int j = 1; j <= width; j++) {
                    x[i][j] = (x0[i][j] + a * (x[i - 1][j] + x[i + 1][j] + x[i][j - 1] + x[i][j + 1])) / c;
                }
            }
        }
    }

    void diffuse( float[][] x, float[][] x0, float diff, float dt) {
        float a = dt * diff * width * height;
        lin_solve( x, x0, a, 1 + 4 * a,2);
    }

    void advect(float[][] d, float[][] d0, float[][] u, float[][] v, float dt) {
        int i0, j0, i1, j1;
        float x, y, s0, t0, s1, t1, dth ;

        dth = dt * height;

        for (int i = 1; i <= height; i++) {
            for (int j = 1; j <= width; j++) {
                x = i - dth * u[i][j];
                y = j - dth * v[i][j];
                if (x < 0.5f) x = 0.5f;
                if (x > height + 0.5f) x = height + 0.5f;
                i0 = (int) x;
                i1 = i0 + 1;
                if (y < 0.5f) y = 0.5f;
                if (y > width+ 0.5f) y = width + 0.5f;
                j0 = (int) y;
                j1 = j0 + 1;
                s1 = x - i0;
                s0 = 1 - s1;
                t1 = y - j0;
                t0 = 1 - t1;
                d[i][j] = s0 * (t0 * d0[i0][j0] + t1 * d0[i0][j1]) +
                        s1 * (t0 * d0[i1][j0] + t1 * d0[i1][j1]);
            }
        }
    }

    void project( float[][] u, float[][] v, float[][] p, float[][] div) {


        for (int i = 1; i <= height; i++) {
            for (int j = 1; j <= width; j++) {
                div[i][j] = -0.5f * (u[i + 1][j] - u[i - 1][j] + v[i][j + 1] - v[i][j - 1]) / height;
                p[i][j] = 0;
            }
        }

        lin_solve( p, div, 1, 4,10);

        for (int i = 1; i <= height; i++) {
            for (int j = 1; j <= width; j++) {
                u[i][j] -= 0.5f * height * (p[i + 1][j] - p[i - 1][j]);
                v[i][j] -= 0.5f * width * (p[i][j + 1] - p[i][j - 1]);
            }
        }
    }

    public void ui_update(float x,float y , float px, float py , float source, float force, int flames) {
        for (int i = 1; i <= height; i++) {
            for (int j = 1; j <= width; j++) {
                x0[i][j] = 0.0f;
                v0[i][j] = 0.0f;
                u0[i][j] = 0.0f;
            }
        }
        for(int i=0;i<flames;i++) {
            int xp=width / (flames+1)*(i+1);
            x0[height-10][xp] = source;
            v0[height-10][xp] = 0;
            u0[height-10][xp] = -force;
        }
        if(x<0) return;
        for(int xx=(int) x-1; xx<(int)x+2; xx++)
            for(int yy=(int) y-1; yy<(int)y+2; yy++) {
                if (xx > 0 && yy > 0 && xx < width && yy < height) {
                    x0[yy][xx] = source;
                    if (px > 0) {
                        v0[yy][xx] = force * (x - px);
                        u0[yy][xx] = force * (y - py);
                    }
                }
            }
    }

    public void dens_step(float diff, float dt) {
        add_source(x, x0, dt);
        diffuse(x0, x, diff, dt);
        advect( x, x0, u, v, dt);
    }

    public void vel_step( float visc, float dt) {
        add_source( u, u0, dt);
        add_source( v, v0, dt);
        diffuse( u0, u, visc, dt);
        diffuse( v0, v, visc, dt);
        project( u0, v0, u, v);
        advect( u, u0, u0, v0, dt);
        advect( v, v0, u0, v0, dt);
        project( u, v, u0, v0);
    }
    int[] bmpmem=null;
    public void fillBitmap(Bitmap bmp) {
        int index=0;
        if (bmpmem==null)
            bmpmem=new int[height*width];
        for (int i = 1; i <= height; i++) {
            for (int j = 1; j <= width; j++) {
                int r = (int) (x[i][j]*255);
                int g = (int) (u[i][j]*-2000)+128;
                int b = (int) (v[i][j]*2000)+128;
                if (r > 255) r = 255;
                if(r<0) r=0;
                if (g > 255) g = 255;
                if(g<0) g=0;
                if (b > 255) b = 255;
                if(b<0) b=0;
                int c = b | (g << 8) | (r << 16) | 0xff000000;
                bmpmem[index++]=c;
            }
        }
        bmp.setPixels( bmpmem,0,width,0,0, width, height);
    }

}
