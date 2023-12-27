#ifndef SPHEREH
#define SPHEREH

#include "vector.h"
#include "ray.h"
#include "hitable.h"
#include "material.h"

class Sphere: public Hitable
{
public:
    __device__ Sphere() {};
    __device__ Sphere(vec3 cen, float r, Material* mat) : center(cen), radius(r), pMat(mat) {};
    __device__ bool hit(ray *pRayIn, float t_min, float t_max, hit_record &pRec) const override;

    vec3 center;
    float radius;
    Material* pMat;
};

__device__ bool Sphere::hit(ray *pRayIn, float tMin, float tMax, hit_record &pRec) const
{
    vec3 oc = pRayIn->origin() - center;
    float a = dot(pRayIn->direction(), pRayIn->direction());
    float b = dot(oc, pRayIn->direction());
    float c = dot(oc, oc) - radius*radius;
    float discrim = b*b - a*c;
    if (discrim > 0)
    {
        // try "minus" quadratic root
        float temp = (-b - sqrt(discrim)) / a;
        if (temp < tMax && temp > tMin)
        {
            pRec.t = temp;
            pRec.p = pRayIn->point_at_parameter(pRec.t);
            pRec.normal = (pRec.p - center) / radius;
            pRec.pMat = pMat;
            return true;
        }
        // try "plus" quadratic root
        temp = (-b + sqrt(discrim)) / a;
        if (temp < tMax && temp > tMin)
        {
            pRec.t = temp;
            pRec.p = pRayIn->point_at_parameter(pRec.t);
            pRec.normal = (pRec.p - center) / radius;
            pRec.pMat = pMat;
            return true;
        }
    }
    return false;
}

#endif
