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
    __device__ bool hit(ray *pRayIn, float tMin, float tMax, hit_record &pRec) const override;

    vec3 center;
    float radius;
    Material* pMat;
};


__device__ bool Sphere::hit(ray *pRayIn, float tMin, float tMax, hit_record &pRec) const
{
    // equation of sphere: x^2 + y^2 + z^2 = r^2
    // therefore, where P = (x,y,z), (P dot P) = r^2
    // given arbitrary center C, this becomes (C-P dot C-P) = r^2
    // we want to know if our ray P(t) = Q + td hits the sphere at any point.
    // rearranging (Q+td dot Q+td) gives the following quadratic:
    //   a = d dot d
    //   b = -2d dot (C-Q)
    //   c = (C-Q) dot (C-Q) - r^2

    vec3 oc = pRayIn->origin() - center;
    // const float a = dot(pRayIn->direction(), pRayIn->direction());
    const float a = pRayIn->direction().squared_length();
    const float b = dot(oc, pRayIn->direction());
    // const float c = dot(oc, oc) - radius*radius;
    const float c = oc.squared_length() - radius*radius;
    const float discrim = b*b - a*c;
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
