CXX=clang++
include Makefile.inc
GENERATOR_SRC=$(HALIDE_SRC_PATH)/tools/GenGen.cpp
GENERATOR_OPTS_RT=-no_asserts-no_bounds_query
#uncomment for autosched (cpu only)
#SCHEDULE=auto_schedule=true -s Adams2019 -p /home/martin/Halide/bin/libautoschedule_adams2019.so
#GPU schedule (not great)
#SCHEDULE=auto_schedule=true -s Li2018 -p /home/martin/Halide/bin/libautoschedule_li2018.so
SCHEDULE=auto_schedule=false

all: demo armeabi-v7a/libnavierstokes_halide.a arm64-v8a/libnavierstokes_halide.a \
x86_64/libnavierstokes_halide.a x86/libnavierstokes_halide.a \
x86_64_gl/libnavierstokes_halide.a arm64-v8a_gl/libnavierstokes_halide.a \
x86_gl/libnavierstokes_halide.a arm64-v8a_cl/libnavierstokes_halide.a \
x86_64_cl/libnavierstokes_halide.a

EMIT=-e o,h,stmt,schedule,assembly

# need to use $$ sometimes in these because it gets expanded twice.
define haliderules
$(1)/libnavierstokes_halide.a : $(1)/halide_dens_step.o $(1)/halide_vel_step.o  $(1)/halide_bitmap.o
	ar r  $(1)/libnavierstokes_halide.a $$^
$(1)/%.o: halide_generator
	./$$< -g $$(*F) -o $(1) target=$(2)$(GENERATOR_OPTS_RT) $(EMIT) $(SCHEDULE)
endef
.SECONDARY: halide_generator
$(eval $(call haliderules,armeabi-v7a,arm-32-android))
$(eval $(call haliderules,arm64-v8a,arm-64-android))
$(eval $(call haliderules,arm64-v8a_gl,arm-64-android-openglcompute))
$(eval $(call haliderules,arm64-v8a_cl,arm-64-android-opencl))
#$(eval $(call haliderules,x86_64,x86-64-android))
$(eval $(call haliderules,x86_64,x86-64-android))
$(eval $(call haliderules,x86_64_gl,x86-64-android-openglcompute))
$(eval $(call haliderules,x86_64_cl,x86-64-android-opencl))
$(eval $(call haliderules,x86,x86-32-android))
$(eval $(call haliderules,x86_gl,x86-32-android-openglcompute))
$(eval $(call haliderules,host,host-profile))
#$(eval $(call haliderules,host,x86-64-linux-avx2-fma-sse41))
$(eval $(call haliderules,host_opencl,host-opencl-profile))
$(eval $(call haliderules,host_cuda,host-cuda-profile))
$(eval $(call haliderules,host_gl,host-openglcompute-profile))
#$(eval $(call haliderules,viz,host-profile-trace_all))

%:: %.cpp
	$(CXX) $(CXXFLAGS) $< $(GENERATOR_SRC) -g $(LIB_HALIDE) -o $@ -lpthread -fno-rtti -ldl -lz -lncurses -rdynamic -O3 $(LDFLAGS) \
	$(LLVM_SHARED_LIBS)
demo: demo.cpp host/libnavierstokes_halide.a
	clang++ -O3  demo.cpp -o demo -lglut -lGL -lGLU -lGLEW -lX11 -ldl -lpthread -lOpenCL host/libnavierstokes_halide.a
demo_c: demo.cpp host/libnavierstokes_halide.a
	clang++ -O3  demo.cpp -o demo_c -lglut -lGL -lGLU -lGLEW -lX11 -ldl -lpthread -lOpenCL host/libnavierstokes_halide.a -DNOUI -DNOHALIDE solver.cpp
demo_opencl: demo.cpp host_opencl/libnavierstokes_halide.a
	clang++ -O3  demo.cpp -o demo_opencl -lglut -lGL -lGLU -lGLEW -lX11 -ldl -lpthread -lOpenCL host_opencl/libnavierstokes_halide.a
demo_cuda: demo.cpp host_cuda/libnavierstokes_halide.a
	clang++ -O3  demo.cpp -o demo_cuda -lglut -lGL -lGLU -lGLEW -lX11 -ldl -lpthread -lOpenCL host_cuda/libnavierstokes_halide.a
demo_gl: demo.cpp host_gl/libnavierstokes_halide.a
	clang++ -O3  demo.cpp -o demo_gl -lglut -lGL -lGLU -lGLEW -lX11 -ldl -lpthread -lOpenCL host_gl/libnavierstokes_halide.a
testdemo: demo
	./demo 384 0.1 0.000006 0 5 500
testdemo_cuda: demo_cuda
	./demo_cuda 384 0.1 0.000006 0 5 500
clean:
	rm -f halide_generator *.a *.o */*.o */*.a */*.h

