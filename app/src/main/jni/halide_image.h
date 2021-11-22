// This header defines a simple Image class which wraps a buffer_t. This is
// useful when interacting with a statically-compiled Halide pipeline emitted by
// Func::compile_to_file, when you do not want to link your processing program
// against Halide.h/libHalide.a.

#ifndef HALIDE_TOOLS_IMAGE_H
#define HALIDE_TOOLS_IMAGE_H

#include <cassert>
#include <cstdlib>
//#include <limits>
//#include <memory>
#include <stdint.h>  // <cstdint> requires C++11

#include "HalideRuntime.h"

namespace Halide {
namespace Tools {

template<typename T>
class Image {
    struct Contents {
        Contents(const halide_buffer_t &b, uint8_t *a) : buf(b), ref_count(1), alloc(a) {
            dims[0] = buf.dim[0];
            dims[1] = buf.dim[1];
            dims[2] = buf.dim[2];
            dims[3] = buf.dim[3];
            delete[] buf.dim;
            buf.dim = dims;
        }

        halide_buffer_t buf;
        halide_dimension_t dims[4];
        int ref_count;
        uint8_t *alloc;

        void dev_free() {
//            halide_device_free(NULL, &buf);
        }

        ~Contents() {
            if (buf.device) {
                dev_free();
            }
            if (alloc != 0)
                delete[] alloc;
//            if(buf.dim!=0)
//		delete[] buf.dim;
        }
    };

    Contents *contents;

    void initialize(int x, int y, int z, int w, bool interleaved) {
        halide_buffer_t buf = {0};
        buf.dimensions = 4;
        buf.dim = new halide_dimension_t[buf.dimensions];
        buf.dim[0].extent = x;
        buf.dim[1].extent = y;
        buf.dim[2].extent = z;
        buf.dim[3].extent = w;
        if (interleaved) {
            buf.dim[0].stride = z;
            buf.dim[1].stride = x * z;
            buf.dim[2].stride = 1;
            buf.dim[3].stride = x * y * z;
        } else {
            buf.dim[0].stride = 1;
            buf.dim[1].stride = x;
            buf.dim[2].stride = x * y;
            buf.dim[3].stride = x * y * z;
        }
        buf.type = halide_type_of<T>();
//        buf.elem_size = sizeof(T);

        size_t size = 1;
        if (x) size *= x;
        if (y) size *= y;
        if (z) size *= z;
        if (w) size *= w;

        uint8_t *ptr = new uint8_t[sizeof(T) * size + 40];
        buf.host = ptr;
        buf.set_host_dirty(false);
        buf.set_device_dirty(false);
        buf.device = 0;
        while ((size_t) buf.host & 0x1f) buf.host++;
        contents = new Contents(buf, ptr);
    }

public:
    typedef T ElemType;

    Image() : contents(NULL) {
    }

    Image(int x, int y = 0, int z = 0, int w = 0, bool interleaved = false) {
        initialize(x, y, z, w, interleaved);
    }
    Image(uint8_t *data, int x, int y, int z, int sx, int sy, int sz) {
        halide_buffer_t buf = {0};
        buf.dimensions = 3;
        buf.dim = new halide_dimension_t[buf.dimensions];
        buf.dim[0].extent = x;
        buf.dim[1].extent = y;
        buf.dim[2].extent = z;
        buf.dim[0].stride = sx;
        buf.dim[1].stride = sy;
        buf.dim[2].stride = sz;
        buf.type = halide_type_of<T>();
        buf.host = data;
        buf.set_host_dirty(false);
        buf.set_device_dirty(false);
        buf.device = 0;
        contents = new Contents(buf, 0);
    }


    Image(const Image &other) : contents(other.contents) {
        if (contents) {
            contents->ref_count++;
        }
    }

    Image(halide_buffer_t &b) {
	    contents = new Contents(b, 0);
    }

    ~Image() {
        if (contents) {
            contents->ref_count--;
            if (contents->ref_count == 0) {
                delete contents;
                contents = NULL;
		
            }
        }
    }

    Image &operator=(const Image &other) {
        Contents *p = other.contents;
        if (p) {
            p->ref_count++;
        }
        if (contents) {
            contents->ref_count--;
            if (contents->ref_count == 0) {
                delete contents;
                contents = NULL;
            }
        }
        contents = p;
        return *this;
    }

    T *data() { return (T*)contents->buf.host; }

    const T *data() const { return (T*)contents->buf.host; }

    void set_host_dirty(bool dirty = true) {
        // If you use data directly, you must also call this so that
        // gpu-side code knows that it needs to copy stuff over.
        contents->buf.set_host_dirty(dirty);
    }

    void copy_to_host() {
        if (contents->buf.device_dirty()) {
            halide_copy_to_host(NULL, &contents->buf);
            contents->buf.set_device_dirty(false);
        }
    }

    void copy_to_device(const struct halide_device_interface *device_interface) {
        if (contents->buf.host_dirty()) {
            // If host
            halide_copy_to_device(NULL, &contents->buf, device_interface);
            contents->buf.set_host_dirty(false);
        }
    }

    void dev_free() {
        assert(!contents->buf.device_dirty());
        contents->dev_free();
    }

//    Image(T vals[]) {
//        initialize(sizeof(vals)/sizeof(T));
//        for (int i = 0; i < sizeof(vals); i++) (*this)(i) = vals[i];
//    }

    /** Make sure you've called copy_to_host before you start
     * accessing pixels directly. */
    T &operator()(int x, int y, int z = 0, int w = 0) {
        T *ptr = (T *)contents->buf.host;
        size_t s0 = contents->buf.dim[0].stride;
        size_t s1 = contents->buf.dim[1].stride;
        size_t s2 = contents->buf.dim[2].stride;
        size_t s3 = contents->buf.dim[3].stride;
        return ptr[s0 * x + s1 * y + s2 * z + s3 * w];
    }

    T &operator()(int x) {
        T *ptr = (T *)contents->buf.host;
        size_t s0 = contents->buf.dim[0].stride;
        return ptr[s0 * x];
    }

    /** Make sure you've called copy_to_host before you start
     * accessing pixels directly */
    const T &operator()(int x, int y, int z = 0, int w = 0) const {
        const T *ptr = (const T *)contents->buf.host;

        size_t s0 = contents->buf.dim[0].stride;
        size_t s1 = contents->buf.dim[1].stride;
        size_t s2 = contents->buf.dim[2].stride;
        size_t s3 = contents->buf.dim[3].stride;
        return ptr[s0 * x + s1 * y + s2 * z + s3 * w];
    }

    const T &operator()(int x) const {
        const T *ptr = (const T *)contents->buf.host;
        size_t s0 = contents->buf.dim[0].stride;
        return ptr[s0 * x];
    }

    operator halide_buffer_t *() const {
        return &(contents->buf);
    }

    int width() const {
        return dimensions() > 0 ? contents->buf.dim[0].extent : 1;
    }

    int height() const {
        return dimensions() > 1 ? contents->buf.dim[1].extent : 1;
    }

    int channels() const {
        return dimensions() > 2 ? contents->buf.dim[2].extent : 1;
    }

    int dimensions() const {
        for (int i = 0; i < 4; i++) {
            if (contents->buf.dim[i].extent == 0) {
                return i;
            }
        }
        return 4;
    }

    int stride(int d) const {
        return contents->buf.dim[d].stride;
    }

//    int min(int d) const {
//        return contents->buf.dim[d].min;
//    }

    int extent(int d) const {
        return contents->buf.dim[d].extent;
    }

};

}  // namespace Tools
}  // namespace Halide

#endif  // HALIDE_TOOLS_IMAGE_H
