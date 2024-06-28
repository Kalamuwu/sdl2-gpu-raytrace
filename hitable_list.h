#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

class HitableList : public Hitable
{
public:
    __device__ HitableList() {}
    __device__ HitableList(Hitable **l, int n) { list = l; list_size = n; }
    __device__ bool hit(ray *pRayIn, float tMin, float tMax, hit_record &pRec) const override;

    Hitable **list;
    int list_size;
};

__device__ bool HitableList::hit(ray *pRayIn, float t_min, float t_max, hit_record &pRec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < list_size; i++)
    {
        if (list[i]->hit(pRayIn, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            pRec = temp_rec;
        }
    }
    return hit_anything;
}

#endif
