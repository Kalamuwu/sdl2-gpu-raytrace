#ifndef HITABLEH
#define HITABLEH

#include "vector.h"
#include "ray.h"

// abstract class for hitable targets.

class Material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    Material *pMat;
};

class Hitable
{
public:
    __device__ virtual bool hit(ray *pRayIn, float tMin, float tMax, hit_record &pRec) const = 0;
};

#endif
