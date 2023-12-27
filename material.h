#ifndef MATERIALH
#define MATERIALH

#include <bits/stdc++.h>
#include <cmath>
#include <curand_kernel.h>

#include "vector.h"
#include "ray.h"
#include "hitable.h"

class Material
{
public:
    __device__ virtual bool scatter(ray *pRay, const hit_record &pRec, vec3 &pAttenuation, curandState *pRandState, bool &isLightSource) const = 0;
};


// Diffuse -- Lambertian diffuse material. Simulates a rough, matte material.
class Diffuse : public Material
{
public:
    __device__ Diffuse(const vec3& a) : albedo(a) {}

    __device__ bool scatter(ray *pRay, const hit_record &pRec, vec3 &pAttenuation, curandState *pRandState, bool &isLightSource) const
    {
        vec3 target = pRec.p + pRec.normal + random_in_unit_sphere(pRandState);
        *pRay = ray(pRec.p, target-pRec.p);
        pAttenuation = albedo;
        isLightSource = false;
        return true;
    }

    vec3 albedo;
};

// Metal -- Reflects incoming rays mirrored across the normal vector.
// Simulates a mirror finish.
class Metal : public Material
{
public:
    __device__ Metal(const vec3& a, const float f) : albedo(a) { fuzz = __saturatef(f); }

    __device__ bool scatter(ray *pRay, const hit_record &pRec, vec3 &pAttenuation, curandState *pRandState, bool &isLightSource) const
    {
        vec3 reflected = reflect(unit_vector(pRay->direction()), pRec.normal);
        *pRay = ray(pRec.p, reflected + fuzz*random_in_unit_sphere(pRandState));
        pAttenuation = albedo;
        isLightSource = false;
        return (dot(pRay->direction(), pRec.normal) > 0);
    }

    vec3 albedo;
    float fuzz;
};


// Emmissive-- Emits more light than it receives
class Emmissive : public Material
{
public:
    __device__ Emmissive(const vec3& a, const float s, const bool c) : albedo(a), strength(s), continueTracing(c) {}

    __device__ bool scatter(ray *pRay, const hit_record &pRec, vec3 &pAttenuation, curandState *pRandState, bool &isLightSource) const
    {
        vec3 target = pRec.p + pRec.normal + random_in_unit_sphere(pRandState);
        *pRay = ray(pRec.p, target-pRec.p);
        pAttenuation = albedo * strength;
        isLightSource = true;
        return continueTracing;
    }

    vec3 albedo;
    float strength;
    bool continueTracing;
};


// Glass - Calculates internal reflections and refractions
class Glass : public Material
{
public:
    __device__ Glass(const vec3& a, const float ri) : albedo(a), ref_idx(ri) {}

    // This was stolen from Peter Shirley's Ray Tracing in One Weekend. Don't
    // ask me how it works. I have a basic understanding but not enough to
    // teach anyone else.
    __device__ bool scatter(ray *pRay, const hit_record &pRec, vec3 &pAttenuation, curandState *pRandState, bool &isLightSource) const
    {
        vec3 outward_normal;
        vec3 reflected = reflect(pRay->direction(), pRec.normal);
        float ni_over_nt;
        pAttenuation = albedo;
        vec3 refracted;
        float cosine;
        if (dot(pRay->direction(), pRec.normal) > 0)
        {
            outward_normal = -pRec.normal;
            ni_over_nt = ref_idx;
            cosine = ref_idx * dot(pRay->direction(), pRec.normal) / pRay->direction().length();
        }
        else
        {
            outward_normal = pRec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(pRay->direction(), pRec.normal) / pRay->direction().length();
        }
        if (this->refract(pRay->direction(), outward_normal, ni_over_nt, refracted))
        {
            float reflect_prob = this->schlick(1-cosine, ref_idx);
            *pRay = ray(pRec.p, (curand_uniform(pRandState) < reflect_prob)? reflected : refracted);
        }
        else *pRay = ray(pRec.p, reflected);
        isLightSource = false;
        return true;
    }

    // Glass has reflectivity that varies with viewing angle. This is a massive
    // ugly equation, but luckily Christophe Schlick came up with this simple
    // polynomial approximation.
    __device__ float schlick(float cosine, float ref_idx) const
    {
        float r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*cosine*cosine*cosine*cosine*cosine;
    }

    // determines whether the viewing angle is steep enough for total internal
    // reflection (zero refraction) and refracts. this is based on snell's law:
    //    n sin(theta) = n' sin(theta')
    // common values:  air=1, glass=1.3-1.7, diamond=2.4
    __device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) const
    {
        vec3 uv = unit_vector(v);
        float dt = dot(uv, n);
        float discriminant = ni_over_nt*ni_over_nt*(1-dt*dt);
        if (discriminant < 1) {
            refracted = ni_over_nt*(uv - n*dt) - n*sqrt(1-discriminant);
            return true;
        }
        else return false;
    }

    vec3 albedo;
    float ref_idx;

};


// Translucent -- Somewhat transparent, tints incoming light
class Translucent : public Material
{
public:
    __device__ Translucent(const vec3& a, const float t, const float s) : albedo(a) {
        translucency = __saturatef(t);
        scattering =   __saturatef(t);
    }

    __device__ bool scatter(ray *pRay, const hit_record &pRec, vec3 &pAttenuation, curandState *pRandState, bool &isLightSource) const
    {
        float cosine = (dot(pRay->direction(), pRec.normal)) / (pRay->direction().length());
        cosine = 0.5f*(cosine+1);
        *pRay = ray(pRec.p, pRec.p + pRay->direction() + scattering*random_in_unit_sphere(pRandState));
        pAttenuation = albedo;
        pAttenuation[0] = fma(pAttenuation[0], cosine, translucency);
        pAttenuation[1] = fma(pAttenuation[1], cosine, translucency);
        pAttenuation[2] = fma(pAttenuation[2], cosine, translucency);
        isLightSource = false;
        return true;
    }

    vec3 albedo;
    float translucency;
    float scattering;
};


// Normals -- Visualizes normals
class Normals : public Material
{
public:
    __device__ Normals() {}

    __device__ bool scatter(ray *pRay, const hit_record &pRec, vec3 &pAttenuation, curandState *pRandState, bool &isLightSource) const
    {
        pAttenuation = pRec.normal;
        isLightSource = true;
        return false;
    }
};

#endif
