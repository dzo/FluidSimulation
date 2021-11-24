package com.example.martin.simulation;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.os.BatteryManager;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

import androidx.preference.PreferenceManager;

import java.util.Random;

/**
 * Created by martin on 19/09/21.
 */

public class SimulationView extends View {

    private static final String TAG = "SimulationView";
    private float mTouchX=-1000;
    private float mTouchY;
    private float mLastX;
    private float mLastY;
    private float mSource;
    private float mForce;
    private long mLasttime =0;
    private int mNative;
    private Paint mPaint=new Paint();
    private int mWidth=0, mHeight=0;
    private Bitmap mBitmap;
    private Simulation mSimulation;
    private float mDiffusion;
    private float mViscosity;
    private float mScale;
    RectF rect=new RectF();
    float mFps=0;
    Context mContext;
    private int mFlames;
    private int mNfps=0;

    public void setNative(int mNative) {
        this.mNative = mNative;
    }

    public void init() {
        int w=(int)(mWidth/ mScale);
        int h=(int)(mHeight/ mScale);
        mBitmap=Bitmap.createBitmap(w,h, Bitmap.Config.ARGB_8888);
        if(mNative>0)
            mSimulation =new NativeSimulation(w,h, mNative);
        else
            mSimulation =new JavaSimulation(w,h);
        mPaint.setTextSize(48);
        mPaint.setColor(Color.WHITE);
        mPaint.setAntiAlias(true);
    }
    public SimulationView(Context context, AttributeSet attrs) {
        super(context,attrs);
        mContext=context;
    }

    public void setParams(float scale, float diff, float visc, float source, float force, int flames) {
        mScale=scale;
        mDiffusion=diff;
        mViscosity=visc;
        mSource=source;
        mForce=force;
        mFlames=flames;
    }

    // only initialise when we know what size the view is
    @Override
    protected void onLayout(boolean changed, int left, int top, int right, int bottom) {
        super.onLayout(changed, left, top, right, bottom);
        Log.i(TAG,"onLayout:"+changed+","+(right - left)+","+(bottom - top));
        if(changed && mWidth==0) {
            mWidth = (right - left);
            mHeight = (bottom - top);
            init();
        }
    }
    String stats="";
    // draw the bitmap
    @Override
    protected void onDraw(Canvas c) {
        super.onDraw(c);
        rect.top=0;
        rect.left=0;
        rect.right=mWidth;
        rect.bottom=mHeight;
        synchronized (mBitmap) {
            c.drawBitmap(mBitmap, null, rect, mPaint);
        }
        c.drawText(stats,20,160,mPaint);
    }

    int getIntPref(int key, int def) {
        return PreferenceManager.getDefaultSharedPreferences(mContext).getInt(mContext.getString(key),Integer.parseInt(mContext.getString(def)));
    }
    // move the simulation on by dt seconds.
    public void update(float dt) {
        if(mSimulation ==null) {
            if(mWidth!=0) init();
            return;
        }
        //dt=0.005f;
    //    mFlames=getIntPref(R.string.flames_key,R.string.default_flames);
        long time=System.currentTimeMillis();
        long time0=System.nanoTime();
        mSimulation.ui_update(mTouchX/mScale,mTouchY/mScale,mLastX/mScale,mLastY/mScale, mSource, mForce, mFlames);
        long time1=System.nanoTime();
        mSimulation.vel_step(mViscosity/10.0f,dt*10);
        long time2=System.nanoTime();
        mSimulation.dens_step(mDiffusion/10.0f,dt*10);
        long time3=System.nanoTime();
        synchronized (mBitmap) {
            mSimulation.fillBitmap(mBitmap);
        }
        long time4=System.nanoTime();
        if(time4!=time1) {
            mFps += 1000000000 / (time4 - time1);
            mNfps++;
        }
        if(time- mLasttime >1000) {
            mLasttime =time;
            Log.i(TAG,"Times:"+(time1-time0)/1000+","+(time2-time1)/1000+","+(time3-time2)/1000+","+(time4-time3)/1000);
            BatteryManager batteryManager = (BatteryManager)mContext.getSystemService(Context.BATTERY_SERVICE);
            float current= 0;
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.LOLLIPOP) {
                current = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)/1000.0f;
            }
            stats = "FPS:"+(int)(mFps/mNfps);
            mNfps=0;
            mFps=0;
            if(current>0) stats+=" Power:"+current+"mA";

        }
    }

    @SuppressLint("ClickableViewAccessibility")
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            mTouchX = event.getX();
            mTouchY = event.getY();
            mLastX=mTouchX;
            mLastY=mTouchY;
            return true;
        }
        if (event.getAction() == MotionEvent.ACTION_UP) {
            mTouchX=-1000;
            return true;
        }

        if (event.getAction() == MotionEvent.ACTION_MOVE) {
            mLastY=mTouchY;
            mLastX=mTouchX;
            mTouchX = event.getX();
            mTouchY = event.getY();
            return true;
        }
        return false;
    }

    public void shutdown() {
        if(mSimulation instanceof NativeSimulation)
            ((NativeSimulation) mSimulation).shutdown();
        mSimulation=null;
    }

}

