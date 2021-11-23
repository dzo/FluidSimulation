package com.example.martin.simulation;

import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.util.Log;
import android.view.Window;

import androidx.appcompat.app.AppCompatActivity;

import static java.lang.Thread.sleep;

import com.example.martin.simulation.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity  {

    private static final String TAG = "Simulation";
    private boolean mPaused=false;
    private Thread mThread;
    private SharedPreferences mPreferences;
    private int mUpdateTime=10;
    private ActivityMainBinding mMainLayout;
    static int mNative=-1;
    private boolean mFixedTimestep;

    String getPref(int key, int def) {
        return mPreferences.getString(getString(key),getString(def));
    }
    int getIntPref(int key, int def) {
        return mPreferences.getInt(getString(key),Integer.parseInt(getString(def)));
    }

    private void init() {
        // if choice of native implementation has changed we need to restart, ugly but necessary
        // because you can't unload a native library.
        int nat=Integer.parseInt(getPref(R.string.halide_key,R.string.halide_default));
        if(nat!=mNative && mNative!=-1 && nat>1){
            System.exit(0);
        }
        mNative=Integer.parseInt(getPref(R.string.halide_key,R.string.halide_default));
        mMainLayout.simulation.setNative(mNative);
        mMainLayout.simulation.setParams(getIntPref(R.string.scale_key,R.string.scale_default),
                getIntPref(R.string.diffusion_key,R.string.diffusion_default)*0.0001f/100f,
                getIntPref(R.string.viscosity_key,R.string.viscosity_default)*0.0001f/100f,
                getIntPref(R.string.source_key,R.string.source_default)*10,
                getIntPref(R.string.force_key,R.string.force_default),
                getIntPref(R.string.flames_key,R.string.default_flames)
                );
        mFixedTimestep=mPreferences.getBoolean(getString(R.string.fixedtimestep_key),false);
        mThread=new Thread(() -> {
            long nanoTime = System.nanoTime() - 10;
            long currentTime;
            long secs = System.currentTimeMillis() / 1000;
            long rendertime = 0;
            while (!mPaused) {
                long newtime = System.nanoTime();
                long dt = newtime - nanoTime;
                nanoTime = newtime;
                currentTime= System.currentTimeMillis();
                if(mFixedTimestep) dt=5000000;
                mMainLayout.simulation.update(dt / 1000000000f);
                newtime = System.currentTimeMillis();
                if (newtime / 1000 > secs) {  // show stats every second for last update
                    Log.d(TAG, "Update took:" + (newtime - currentTime) + " dt=" + dt/1000000f);
                    secs = newtime / 1000;
                }
                if (newtime - rendertime > 16) {
                    mMainLayout.simulation.postInvalidate();
                    rendertime = newtime;
                }
                long waittime = mUpdateTime - (newtime - currentTime);
                if (waittime > 0) {
                    try {
                        sleep(waittime);
                    } catch (InterruptedException ignored) {

                    }
                }
            }
        },"UpdateThread");

        mThread.start();
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mMainLayout = ActivityMainBinding.inflate(getLayoutInflater());
        mPreferences= PreferenceManager.getDefaultSharedPreferences(this);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        setContentView(mMainLayout.getRoot());
        mMainLayout.simulation.setFocusable(true);
        mUpdateTime=getIntPref(R.string.updatetime_key,R.string.default_updatetime);
        init();
        findViewById(R.id.menu).setOnClickListener(view -> {
            startActivity(new Intent(MainActivity.this,SettingsActivity.class));
         });
    }

    @Override
    public void onPause() {
        super.onPause();
        mPaused=true;
        try {
            mThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        mThread=null;
        mMainLayout.simulation.shutdown();
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.i(TAG,"onResume");
        if(mPaused) {
            Log.i(TAG,"onResumePaused");
            mPaused=false;
            init();
        }
    }
}
