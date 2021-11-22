#include <Halide.h>
using namespace Halide;
using namespace std;
bool gpu=false;
bool auto_sch=false;

Var x("x"), y("y"),xi("xi"), yi("yi"), yo("yo"),xo("xo");
Func add_source(Func in, Func s, Expr dt ) {
    Func f{"add_source"};
    f(x,y) = in(x,y) + dt*s(x,y);
    return f;
}

void gpu_schedule(vector <Func> v) {
    for(Func f:v)
       f.compute_root().gpu_tile(x,y, xo,yo,xi, yi, 16,16);
}

void cpu_schedule(vector <Func> v) {
    for(Func f:v) {
        f.tile(x, y, xi, yi, 16, 16);
        f.compute_root().parallel(y).vectorize(xi);
    }
}

void good_schedule(vector <Func> v) {
    if(auto_sch)
        return;
    if (gpu)
        gpu_schedule(v);
    else
        cpu_schedule(v);
}

Func convolve(Func in, Expr w, Expr h) {
    Func fin=BoundaryConditions::constant_exterior(in, 0, 1, w, 1, h);
    Func fx("fx"),fy("fy");

    const float H=sqrt(0.7667627574610649);
    const float T[]={6.25750013e-02f*H, 1.23582761e-01f*H, 2.28085967e-01f*H, 4.08342503e-01f*H, 7.22410879e-01f*H};

    fx(x,y)=T[0]*fin(x,y-4)+T[1]*fin(x,y-3)+T[2]*fin(x,y-2)+T[3]*fin(x,y-1)+T[4]*fin(x,y)+
            T[0]*fin(x,y+4)+T[1]*fin(x,y+3)+T[2]*fin(x,y+2)+T[3]*fin(x,y+1);
    fy(x,y)=T[0]*fx(x-4,y)+T[1]*fx(x-3,y)+T[2]*fx(x-2,y)+T[3]*fx(x-1,y)+T[4]*fx(x,y)+
            T[0]*fx(x+4,y)+T[1]*fx(x+3,y)+T[2]*fx(x+2,y)+T[3]*fx(x+1,y);

    if(!auto_sch) {
        if (gpu) {
            gpu_schedule({fx, fy});
        } else {
            fx.compute_root()
                    .split(y, y, yi, 16)
                    .parallel(y)
                    .vectorize(x, 4);
            fy.compute_root()
                    .split(y, y, yi, 16)
                    .parallel(y)
                    .vectorize(x, 4);
        }
    }
    return fy;
}
Func lin_solve(Func in, Func x0, Expr a, Expr c, Expr w, Expr h, int num_steps= 4) {
    Func x1=BoundaryConditions::constant_exterior(x0, 0, 1, w, 1, h);
    Func fin=BoundaryConditions::constant_exterior(in, 0, 1, w, 1, h);
    Expr invc=1.0f/c;

    Func f[num_steps+1];
    f[0]=fin;
    // uses Jacobi rather than Gauss-Seidel so converges a bit slower
    for(int k=1 ; k<=num_steps ; k++) {

        f[k](x, y) = invc*(x1(x, y) + a *(f[k - 1](x - 1, y)
                                        + f[k - 1](x + 1, y)
                                        + f[k - 1](x, y - 1)
                                        + f[k - 1](x, y + 1)));
        if(k%2==0 || k==num_steps) {
            if(!auto_sch) {
                if (gpu) {
                    gpu_schedule({f[k]});
                } else {
                    f[k].tile(x, y, xi, yi, 16, 16);
                    f[k].compute_root().parallel(y).vectorize(xi, 4);
                    f[k - 1].compute_at(f[k], y).vectorize(x, 4);
                }
            }
        }
    }
    return f[num_steps];
}

Func diffuse(Func dens, Func dens0, Expr diff, Expr dt, Expr w, Expr h ) {
    Expr a = dt*diff*w*h/2;
    Func out;
    //out(x,y)=select(a==0,BoundaryConditions::constant_exterior(dens0, 0, 1, w, 1, h)(x,y),lin_solve(dens, dens0, a, 1.0f+4.0f*a, w, h,2)(x,y));
    out= lin_solve(dens, dens0, a, 1.0f + 4.0f * a, w, h, 2);
    return out;
}

const float H1[5] = {0.1507f, 0.6836f, 1.0334f, 0.6836f, 0.1507f};
const float H2	= 0.0270f;
const float G[3]  = {0.0312f, 0.7753f, 0.0312f};

Func upscale(Func current, Func prev) {
    Func g_filt,out;
    Expr e=0;
    for(int i=-1;i<2;i++)
        for(int j=-1;j<2;j++)
            e=e+current(x+i,y+j)*G[i+1]*G[j+1];
    g_filt(x,y)=e;
    e=0;
    int nn[2][2]={{1,0},{0,0}};
    vector <Expr> e1={0,0,0,0};
    for(int ii=0;ii<2;ii++)
        for(int jj=0;jj<2;jj++)
            for(int i=-2;i<3;i++)
                for(int j=-2;j<3;j++)
                    e1[jj+ii*2]+=H1[i+2]*H1[j+2]*prev(x/2+j/2,y/2+i/2)*nn[(ii+i+2)%2][(jj+j+2)%2];
    out(x,y)=g_filt(x,y)+H2*mux(x%2+y%2*2,e1);
    return out;
}

Func downscale(Func in) {
    Expr e=0;
    Func out;
    for(int i=-2;i<3;i++)
        for(int j=-2;j<3;j++)
            e=e+H1[i+2]*H1[i+2]*in(x*2+j,y*2+i);
    out(x,y)=e;
    return out;
}

Func conv_pyr(Func in) {
    int LEVELS=5;
    Func p[LEVELS];
    Func q[LEVELS];
    p[0]=in;
    for(int i=1;i<LEVELS;i++) {
        p[i] = downscale(p[i - 1]);
        good_schedule({p[i]});
    }
    for(int i=LEVELS-2;i>=0;i--) {
        q[i] = upscale(p[i], p[i + 1]);
        good_schedule({q[i]});
    }
    return q[0];
}

Func advect (Func d0, Func u, Func v, Expr dt, Expr w, Expr h)
{
    Func advected{"advected"};

    Expr dt0 = dt*h;

    Expr xx = clamp(x-dt0*v(x,y), 0.5f, w+0.5f);
    Expr yy = clamp(y-dt0*u(x,y), 0.5f, h+0.5f);

    Expr i0=cast<int>(xx);
    Expr j0=cast<int>(yy);
    Expr i1=i0+1;
    Expr j1=j0+1;
    Expr s1 = xx-i0;
    Expr t1 = yy-j0;
    advected(x,y) = lerp(lerp(d0(i0,j0),d0(i0,j1),t1),lerp(d0(i1,j0),d0(i1,j1),t1),s1);
    return advected;
}

void project (Func u, Func v, Func uu, Func vv, Expr w, Expr h )
{
    Func div("div");
    Expr m=-1.0f/h;
    div(x,y) = m*(v(x+1,y)-v(x-1,y)+u(x,y+1)-u(x,y-1));
    good_schedule({div});
    Func p =convolve(div, w, h);
    vv(x,y) = v(x,y) - 0.5f*w*(p(x+1,y)-p(x-1,y));
    uu(x,y) = u(x,y) - 0.5f*h*(p(x,y+1)-p(x,y-1));
}

class DensStepGenerator : public Halide::Generator<DensStepGenerator> {
public:
    // Input parameters
    Input <Buffer<float>> dens{"dens", 2};
    Input <Buffer<float>> dens_prev{"dens_prev", 2};
    Input <Buffer<float>> u{"u", 2};
    Input <Buffer<float>> v{"v", 2};
    Input <float> diff{"diff"};
    Input <float> dt{"dt"};

    Output <Buffer<float>> output{"output", 2};
    Func src_added;
    Func diffused;
    void generate() {
        gpu = get_target().has_gpu_feature() || get_target().has_feature(Target::OpenGLCompute);
        auto_sch = auto_schedule;
        Expr w = dens.width() - 2;
        Expr h = dens.height() - 2;
        src_added = add_source(dens, dens_prev, dt);
        diffused = diffuse(dens_prev, src_added, diff, dt, w, h);
        output(x, y) = advect(diffused, u, v, dt, w, h)(x, y);
    }

    void schedule() {
        int parallel_task_size = 8;
        int vector_width = 4;
        if (auto_schedule) {
            u.set_estimates({{0, 512},
                             {0, 786}});
            v.set_estimates({{0, 512},
                             {0, 786}});
            dens.set_estimates({{0, 512},
                                {0, 786}});
            dens_prev.set_estimates({{0, 512},
                                     {0, 786}});
            output.set_estimates({{0, 512},
                                  {0, 786}});
            dt.set_estimate(0.1);
            diff.set_estimate(0.00001);
        } else {
            good_schedule({output});
        }
    }

};

class VelStepGenerator : public Halide::Generator<VelStepGenerator> {
public:
    // Input parameters
    Input <Buffer<float>> u{"u", 2};
    Input <Buffer<float>> v{"v", 2};
    Input <Buffer<float>> u0{"u0", 2};
    Input <Buffer<float>> v0{"v0", 2};
    Input<float> visc{"visc"};
    Input<float> dt{"dt"};

    Output<Buffer<float>> outputu{"outputu", 2};
    Output<Buffer<float>> outputv{"outputv", 2};

    Func uu, vv, diffU, diffV, projected;
    Func au0, av0;
    Func adU, adV;
    Func op;

    void generate() {
        gpu = get_target().has_gpu_feature() || get_target().has_feature(Target::OpenGLCompute);
        auto_sch = auto_schedule;
        Expr w = u.width() - 2;
        Expr h = u.height() - 2;
        uu = add_source(u, u0, dt);
        diffU = diffuse(u0, uu, visc, dt, w, h);
        vv = add_source(v, v0, dt);
        diffV = diffuse(v0, vv, visc, dt, w, h);
        project(diffU, diffV, au0, av0, w, h);
        adU = advect(au0, au0, av0, dt, w, h);
        adV = advect(av0, au0, av0, dt, w, h);
        project(adU, adV, outputu, outputv, w, h);
    }

    void schedule() {
        if (auto_schedule) {
            u.set_estimates({{0, 512},
                             {0, 786}});
            v.set_estimates({{0, 512},
                             {0, 786}});
            u0.set_estimates({{0, 512},
                              {0, 786}});
            v0.set_estimates({{0, 512},
                              {0, 786}});
            outputu.set_estimates({{0, 512},
                                   {0, 786}});
            outputv.set_estimates({{0, 512},
                                   {0, 786}});
            dt.set_estimate(0.1);
            visc.set_estimate(0.00001);
        } else {
            good_schedule({uu, vv, au0, av0, adU, adV, outputu, outputv});
            if(!gpu) {
                adU.compute_with(adV, x);
                uu.compute_with(vv, x);
                outputu.compute_with(outputv, x);
            }
        }
    }
};


class BitmapGenerator : public Halide::Generator<BitmapGenerator> {
public:
    Input <Buffer<float>> d{"d", 2};
    Input <Buffer<float>> u{"u", 2};
    Input <Buffer<float>> v{"v", 2};
    Output<Buffer<unsigned>> output{"output", 2};

    void generate() {
        gpu = get_target().has_gpu_feature() || get_target().has_feature(Target::OpenGLCompute);
        Expr r = clamp(cast<int>(d(x + 1, y + 1) * 255), 0, 255);
        Expr g = clamp(cast<int>(u(x + 1, y + 1) * -2000) + 128, 0, 255);
        Expr b = clamp(cast<int>(v(x + 1, y + 1) * 2000) + 128, 0, 255);
        output(x, y) = cast<unsigned>(r | (g << 8) | (b << 16) | 0xff000000);
    }

    void schedule() {
        if (auto_schedule) {
          d.set_estimates({{0, 512},
                             {0, 786}});
          u.set_estimates({{0, 512},
                             {0, 786}});
          v.set_estimates({{0, 512},
                             {0, 786}});
          output.set_estimates({{0, 512},
                             {0, 786}});
        } else
            good_schedule({output});
    }
};
HALIDE_REGISTER_GENERATOR(DensStepGenerator,halide_dens_step)
HALIDE_REGISTER_GENERATOR(VelStepGenerator,halide_vel_step)
HALIDE_REGISTER_GENERATOR(BitmapGenerator,halide_bitmap)


