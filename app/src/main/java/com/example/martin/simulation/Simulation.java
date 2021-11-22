package com.example.martin.simulation;

import android.graphics.Bitmap;

public interface Simulation {
     void ui_update(float x, float y, float px, float py, float source, float force, int flames);
     void dens_step(float diff, float dt);
     void vel_step(float visc, float dt);
     void fillBitmap(Bitmap b);
}
