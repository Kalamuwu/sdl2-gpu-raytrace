#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <curand_kernel.h>

class vec3
{
public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline bool operator==(const vec3& v2) const;
    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; }


    __host__ __device__ inline vec3& operator+=(const vec3 &v2);
    __host__ __device__ inline vec3& operator-=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const vec3 &v2);
    __host__ __device__ inline vec3& operator/=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    __host__ __device__ inline float length() const {
        return sqrt( e[0]*e[0] + e[1]*e[1] + e[2]*e[2] );
    }
    __host__ __device__ inline float squared_length() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
    __host__ __device__ inline void make_unit_vector();
    __host__ __device__ inline void make_pos_unit_vector();

    __host__ __device__ inline vec3 clamp(const float min, const float max);
    __host__ __device__ inline vec3 clamp(const vec3& min, const vec3& max);

    // CUDA devices have an efficient instruction to clamp a float to the range
    // +0...1, however x86 does not
    __device__ inline vec3 clampTo01();

    __host__ __device__ inline vec3 modulo(const float t);
    __host__ __device__ inline vec3 modulo(const vec3& v);

    float e[3];
};

__host__ __device__ inline bool vec3::operator==(const vec3& v2) const
{
    return e[0]==v2.e[0] && e[1]==v2.e[1] && e[2]==v2.e[2];
}

__host__ inline std::istream& operator>>(std::istream &is, vec3 &t)
{
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

__host__ inline std::ostream& operator<<(std::ostream &os, const vec3 &t)
{
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}
__host__ __device__ inline vec3 operator*(const vec3 &v, float t)
{
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
    return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2)
{
    return fma(v1.e[0], v2.e[0], fma(v1.e[1], v2.e[1], v1.e[2]*v2.e[2]));
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
    const vec3 v3 = -1*v1;
    return vec3(
           fma(v1.e[1], v2.e[2], v3.e[2]*v2.e[1]),
           fma(v1.e[2], v2.e[0], v3.e[0]*v2.e[2]),
           fma(v1.e[0], v2.e[1], v3.e[1]*v2.e[0])
    );
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3 &v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(float t)
{
    float k = 1.0f/t;
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) { return v / v.length(); }

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) { return v - 2*dot(v,n)*n; }



__host__ inline vec3 random_in_unit_sphere()
{
    vec3 p;
    do {
        // pick random point in unit cube
        p = 2.0f*vec3(drand48(),drand48(),drand48()) - vec3(1,1,1);
    // reject while not in unit sphere
    } while (p.squared_length() >= 1);
    return p;
}

__device__ inline vec3 random_in_unit_sphere(curandState *pRandState)
{
    vec3 p;
    do {
        // pick random point in unit cube
        p = 2.0f*vec3(curand_uniform(pRandState),curand_uniform(pRandState),curand_uniform(pRandState)) - vec3(1,1,1);
    // reject while not in unit sphere
    } while (p.squared_length() >= 1);
    return p;
}

__host__ __device__ inline void vec3::make_unit_vector()
{
    // rsqrtf(a) == 1.0f / sqrt(a)
    float k = rsqrtf( e[0]*e[0] + e[1]*e[1] + e[2]*e[2] );
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline void vec3::make_pos_unit_vector()
{
    make_unit_vector();
    e[0] = 0.5f * (e[0]+1);
    e[1] = 0.5f * (e[1]+1);
    e[2] = 0.5f * (e[2]+1);
}

__host__ __device__ inline vec3 vec3::clamp(const float min, const float max)
{
    const float a = (e[0] < min)? min : e[0];
    const float b = (e[1] < min)? min : e[1];
    const float c = (e[2] < min)? min : e[2];
    return vec3(
        (a > max? max : a),
        (b > max? max : b),
        (c > max? max : c)
    );
}

__host__ __device__ inline vec3 vec3::clamp(const vec3& min, const vec3& max)
{
    const float a = (e[0] < min.e[0])? min.e[0] : e[0];
    const float b = (e[1] < min.e[1])? min.e[1] : e[1];
    const float c = (e[2] < min.e[2])? min.e[2] : e[2];
    return vec3(
        (a > max.e[0]? max.e[0] : a),
        (b > max.e[1]? max.e[1] : b),
        (c > max.e[2]? max.e[2] : c)
    );
}

__device__ inline vec3 vec3::clampTo01()
{
    return vec3(
        __saturatef(e[0]),
        __saturatef(e[1]),
        __saturatef(e[2])
    );
}

__host__ __device__ inline vec3 vec3::modulo(const float t)
{
    return vec3(fmod(e[0],t), fmod(e[1],t), fmod(e[2],t));
}

__host__ __device__ inline vec3 vec3::modulo(const vec3& v)
{
    return vec3(fmod(e[0], v.e[0]), fmod(e[1], v.e[1]), fmod(e[2], v.e[2]));
}

#endif
