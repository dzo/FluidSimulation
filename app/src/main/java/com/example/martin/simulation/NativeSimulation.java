package com.example.martin.simulation;

import android.graphics.Bitmap;

public class NativeSimulation implements Simulation {
    public
    NativeSimulation(int w, int h, int halide) {
        try {
            if (halide == 2 || halide == 1) System.loadLibrary("navierstokes");
            if (halide == 3) System.loadLibrary("navierstokes_gl");
            if (halide == 4) System.loadLibrary("navierstokes_cl");
        } catch (UnsatisfiedLinkError e) {
            System.loadLibrary("navierstokes"); // should always work.
        }
        init(w,h, halide);
    }
    public native void init(int w, int h, int halide);
    public native void ui_update(float x,float y , float px, float py , float source, float force, int flames);
    public native void dens_step(float diff, float dt);
    public native void vel_step( float visc, float dt);
    public native void fillBitmap(Bitmap b);
    public native void shutdown();
}
