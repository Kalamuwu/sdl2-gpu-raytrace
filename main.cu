#include <curand_kernel.h>
#include <time.h>
#include <SDL2/SDL.h>

#include "macros.h"

#include "screen.h"
#include "vector.h"
#include "ray.h"
#include "camera.h"
#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"
#include "material.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error " << static_cast<unsigned int>(result) << " " << cudaGetErrorString(result)
                  << " at " << file << ":" << line << " '" << func << "'\n";
        cudaDeviceReset();
        exit(static_cast<unsigned int>(result));
    }
}

// Kernels for memory management on the device. 'd' prefix represents
// device-only data.
#define NUM_WORLD_ELEMENTS 9
__global__ void alloc_world(Hitable **dList, Hitable **dWorld, Camera **dCamera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) // only run once
    {
        dList[0] = new Sphere(vec3( 0,   100.6, -2  ), 100, new Diffuse(   vec3(0.3, 0.5, 0.7)                ));
        dList[1] = new Sphere(vec3( 0,     0,   -2  ), 0.5, new Diffuse(   vec3(0.8, 0.3, 0.3)                ));
        dList[2] = new Sphere(vec3( 2.6,  -1.4, -1.7), 0.7, new Metal(     vec3(0.9, 0.5, 0.2),   0.2         ));
        dList[3] = new Sphere(vec3( 1,     0,   -2  ), 0.4, new Metal(     vec3(0.3, 0.4, 0.9),   0.05        ));
        dList[4] = new Sphere(vec3(-0.3,   0.1, -1  ), 0.3, new Glass(     vec3(0.5, 1.0, 0.6),   0.9         ));
        dList[5] = new Sphere(vec3( 0,     0.2,  1  ), 0.3, new Glass(     vec3(0.8, 0.2, 0.3),   0.0         ));
        dList[6] = new Sphere(vec3(-1,    -0.3, -1.2), 0.2, new Emmissive( vec3(0.3, 0.2, 0.0),   9.0,  false ));
        dList[7] = new Sphere(vec3( 0.3,  -0.5, -1.1), 0.2, new Emmissive( vec3(0.0, 0.1, 0.9),  10.0,  false ));
        dList[8] = new Sphere(vec3( 0,    -5.0, -4  ), 2.0, new Emmissive( vec3(1.0, 1.0, 1.0),   1.0,  false ));
        *(dWorld)  = new HitableList(dList, NUM_WORLD_ELEMENTS);
        *(dCamera) = new Camera(
            vec3(-1,0,2),
            vec3(0,0,-1),
            vec3(0,1,0),
            80,
            (float)WINDOW_WIDTH/WINDOW_HEIGHT
        );
    }
}
__global__ void free_world(Hitable **dList, Hitable **dWorld, Camera **dCamera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) // only run once
    {
        for (int i = 0; i < NUM_WORLD_ELEMENTS; i++)
        {
            delete ((Sphere *)dList[i])->pMat;
            delete dList[i];
        }
        delete *dWorld;
        delete *dCamera;
    }
}
// Sets up a random number state for each thread. This cannot be a
// global RNG, because each thread will call it simultaneously, and
// all recieve the same number.
__global__ void init_rands(curandState *pRandStates)
{
    float i = threadIdx.x + blockIdx.x * blockDim.x;
    float j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= WINDOW_WIDTH) || (j >= WINDOW_HEIGHT)) return;
    uint32_t pixel_index = (uint32_t)j*WINDOW_WIDTH + (uint32_t)i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(0xE621, pixel_index, 0, &pRandStates[pixel_index]);
}

__device__ vec3 color(ray *r, Hitable **world, curandState *pRandState)
{
    vec3 runningAttenuation = vec3(1,1,1);
    hit_record rec;
    vec3 attenuation;
    for (int i = 0; i < MAX_NUM_REFLECTIONS; i++)  // do MAX_NUM_REFLECTIONS reflections
    {
        bool isLightSource = false;
        if ((*world)->hit(r, 0.001f, FLT_MAX, rec))
        {
            if (rec.pMat->scatter(r, rec, attenuation, pRandState, isLightSource))
                runningAttenuation *= attenuation;
            else return runningAttenuation * attenuation * isLightSource;
        }
        else return runningAttenuation * SKYBOX_COLOR;
        // // lerp white...blue and multiply by attenuation
        // vec3 unit_direction = unit_vector(r.direction());
        // float t = 0.5f*(unit_direction.y()+1.0f);
        // return runningAttenuation * ((1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0));
    }

    // exceeded recursion
    return vec3(0,0,0);
}

// Render kernel -- steps through raytracing a pixel, NUM_ALIAS_STEPS
// times, and averages their values. This averaging step (1) smoothes
// out render artifacts, and (2) achieves an antialiasing effect.
__global__ void render(uint32_t *pFrameBuffer, Camera **pCam, Hitable **world, curandState *pRandStates)
{
    // __fmaf_rz(float a, float b, float c) == a*b+c
    float i = __fmaf_rz(blockIdx.x, blockDim.x, threadIdx.x);
    float j = __fmaf_rz(blockIdx.y, blockDim.y, threadIdx.y);
    if ((i >= WINDOW_WIDTH) || (j >= WINDOW_HEIGHT)) return;
    uint32_t pixel_index = (uint32_t)j*WINDOW_WIDTH + (uint32_t)i;
    curandState pLocalRandState = pRandStates[pixel_index];

    vec3 col(0,0,0);
    // for (int iter = 0; iter < NUM_ALIAS_STEPS; iter++)
    // {
    //     // add drand48() for a slight randomization to the direction.
    //     // this non-uniformity is what achieves the above benefits.
    //     // note: for CUDA, drand48() is not available. instead,
    //     // curand_uniform(curandState*) is equivalent.
    //     float u = (i + curand_uniform(&pLocalRandState)) * (1.0f / WINDOW_WIDTH);
    //     float v = (j + curand_uniform(&pLocalRandState)) * (1.0f / WINDOW_HEIGHT);
    //
    //     ray r = (*pCam)->getRay(u, v);
    //
    //     col += color(&r, world, &pLocalRandState);
    // }
    #define DUFF_DEVICE_16(aAction) \
    do { \
    int times = (NUM_ALIAS_STEPS + 15) >> 4; \
    switch (NUM_ALIAS_STEPS & 15) { \
    case 0: do { aAction; \
    case 15:     aAction; \
    case 14:     aAction; \
    case 13:     aAction; \
    case 12:     aAction; \
    case 11:     aAction; \
    case 10:     aAction; \
    case 9:      aAction; \
    case 8:      aAction; \
    case 7:      aAction; \
    case 6:      aAction; \
    case 5:      aAction; \
    case 4:      aAction; \
    case 3:      aAction; \
    case 2:      aAction; \
    case 1:      aAction; } while (--times > 0); \
    } } while (0)

    float u;
    float v;
    ray r;

    DUFF_DEVICE_16(
        u = (i + curand_uniform(&pLocalRandState)) * (1.0f / WINDOW_WIDTH);
        v = (j + curand_uniform(&pLocalRandState)) * (1.0f / WINDOW_HEIGHT);
        r = (*pCam)->getRay(u, v);
        col += color(&r, world, &pLocalRandState);
    );

    col *= (1.0f / NUM_ALIAS_STEPS);

    uint8_t buffer[4];
    #if __BYTE_ORDER == __LITTLE_ENDIAN
        // A square root is present because SDL assumes the image is
        // gamma-corrected. It is not. This is corrected by raising
        // the color to the power of 1/gamma. To simplify this math,
        // gamma=2 is used. The builtin CUDA __saturatef function is
        // an efficient single-tick call to clamp a float from 0...1;
        // note, x86 does not have an instruction like this. Clamping
        // the colors to 0-1 is important because light sources can
        // go above that, and cause overflow issues and really
        // strange visual glitches. The rest of this mess scales 0..1
        // --> 0..255, and converts to uint8_t to chain together.
        buffer[3] = uint8_t(__saturatef( sqrt(col[0]) ) * 255);
        buffer[2] = uint8_t(__saturatef( sqrt(col[1]) ) * 255);
        buffer[1] = uint8_t(__saturatef( sqrt(col[2]) ) * 255);
        buffer[0] = 0xFF;
    #elif __BYTE_ORDER == __BIG_ENDIAN
        // Same note about square roots as above.
        buffer[1] = uint8_t(__saturatef( sqrt(col[0]) ) * 255);
        buffer[2] = uint8_t(__saturatef( sqrt(col[1]) ) * 255);
        buffer[3] = uint8_t(__saturatef( sqrt(col[2]) ) * 255);
        buffer[4] = 0xFF;
    #else
    # error "Please fix <bits/endian.h>"
    #endif
    pFrameBuffer[pixel_index] = *((uint32_t *)buffer);
}

int main()
{
    clock_t setup_start, render_start, teardown_start, teardown_stop;
    setup_start = clock();
    // Thread and block setup
    // Larger blocks may run faster, since more threads can be used;
    // however, too large a block size will mean that cores will be
    // doing dissimilar operations, greatly reducing performance.
    // GPUs are built and refined for multiple similar operations at
    // the same time, not multiple diverse operations, per thread.
    dim3 blockDims(32,32);
    dim3 gridDims(float(WINDOW_WIDTH) / blockDims.x + 1, float(WINDOW_HEIGHT) / blockDims.y + 1);
    int num_pixels = WINDOW_WIDTH * WINDOW_HEIGHT;

    // Allocate framebuffer and send to cuda
    size_t fbSize = num_pixels*sizeof(uint32_t);
    uint32_t *pFrameBuffer;
    checkCudaErrors(cudaMallocManaged((void**)&pFrameBuffer, fbSize));
    // Allocate random state storage
    curandState *dRandStates;
    checkCudaErrors(cudaMalloc((void **)&dRandStates, num_pixels*sizeof(curandState)));
    // Allocate and set up world objects
    Hitable **dList;
    Hitable **dWorld;
    checkCudaErrors(cudaMalloc((void **)&dList,  sizeof(Hitable *)*NUM_WORLD_ELEMENTS));
    checkCudaErrors(cudaMalloc((void **)&dWorld, sizeof(Hitable *)));
    // Allocate camera
    Camera **dCamera;
    checkCudaErrors(cudaMalloc((void **)&dCamera, sizeof(Camera *)));

    // Device-side allocate world
    alloc_world<<<1,1>>>(dList, dWorld, dCamera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // Initialize RNG and per-thread state
    init_rands<<<gridDims, blockDims>>>(dRandStates);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Set up SDL2 window
    SDL_Event e;
    Screen screen(WINDOW_WIDTH, WINDOW_HEIGHT, 1);
    screen.show();
    // change framebuffer to &fb, to speed up render time
    free(screen.pTextureBuffer);
    screen.pTextureBuffer = pFrameBuffer;


    // Render
    render_start = clock();
    render<<<gridDims, blockDims>>>(pFrameBuffer, dCamera, dWorld, dRandStates);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    teardown_start = clock();
    screen.show();

    // free objects, gpu side
    free_world<<<1,1>>>(dList, dWorld, dCamera);
    checkCudaErrors(cudaGetLastError());
    // free object pointers, gpu side
    checkCudaErrors(cudaFree(dCamera));
    checkCudaErrors(cudaFree(dWorld));
    checkCudaErrors(cudaFree(dList));
    checkCudaErrors(cudaFree(dRandStates));
    checkCudaErrors(cudaFree(pFrameBuffer));

    teardown_stop = clock();
    float setup_seconds =    ((float)(render_start - setup_start))     / CLOCKS_PER_SEC;
    float render_seconds =   ((float)(teardown_start - render_start))  / CLOCKS_PER_SEC;
    float teardown_seconds = ((float)(teardown_stop - teardown_start)) / CLOCKS_PER_SEC;
    // return vec3(setup_seconds, render_seconds, teardown_seconds);
    printf("Render complete.\n");
    printf("Setup took:    %.5f seconds.\n", setup_seconds);
    printf("Render took:   %.5f seconds.\n", render_seconds);
    printf("Teardown took: %.5f seconds.\n", teardown_seconds);
    screen.save("render.png");

    while (1)
    {
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
            {
                goto quit;
            }
        }
        SDL_Delay(10);
    }

    // // Output FB as Image
    // std::cout << "P3\n" << WINDOW_WIDTH << " " << WINDOW_HEIGHT << "\n255\n";
    // for (int j = 0; j < WINDOW_HEIGHT; j++) {
    //     for (int i = 0; i < WINDOW_WIDTH; i++) {
    //         int c = pFrameBuffer[j*WINDOW_WIDTH + i];
    //         #if __BYTE_ORDER == __LITTLE_ENDIAN
    //             int r = 0xFF & (c>>24);
    //             int g = 0xFF & (c>>16);
    //             int b = 0xFF & (c>>8);
    //             // pFrameBuffer[pixel_index] = (ir<<24) | (ig<<16) | (ib<<8) | 0xFF;
    //         #elif __BYTE_ORDER == __BIG_ENDIAN
    //             int r = 0xFF & (c>>0);
    //             int g = 0xFF & (c>>8);
    //             int b = 0xFF & (c>>16);
    //             // pFrameBuffer[pixel_index] = (0xFF<<24) | (ib<<16) | (ig<<8) | ir;
    //         #else
    //         # error "Please fix <bits/endian.h>"
    //         #endif
    //         std::cout << r << " " << g << " " << b << "\n";
    //     }
    // }


quit:
    screen.quit(false);
    SDL_Quit();
    exit(0);
}

// __host__ float custommin(float a, float b)
// {
//     if (a<b) return a;
//     return b;
// }
//
// __host__ float custommax(float a, float b)
// {
//     if (a>b) return a;
//     return b;
// }
//
// __host__ int main()
// {
//     vec3 vtotal(0,0,0);
//     vec3 vmin(FLT_MAX, FLT_MAX, FLT_MAX);
//     vec3 vmax(0,0,0);
//
//     for (int i = 0; i < 10; i++)
//     {
//         printf("Run %d:  ", i);
//         fflush(stdout);
//         vec3 time = _main();
//         printf("%.2f %.2f %.2f\n", time[0], time[1], time[2]);
//         vtotal += time;
//         for (int j = 0; j < 3; j++)
//         {
//             vmin[0] = custommin(vmin[0], time[0]);
//             vmin[1] = custommin(vmin[1], time[1]);
//             vmin[2] = custommin(vmin[2], time[2]);
//             vmax[0] = custommax(vmax[0], time[0]);
//             vmax[1] = custommax(vmax[1], time[1]);
//             vmax[2] = custommax(vmax[2], time[2]);
//         }
//     }
//     vec3 vaverage = vtotal / 10;
//
//     printf("\n---\n\n");
//     printf("Ttl:  %3.4f %3.4f %3.4f\n", vtotal[0],   vtotal[1],   vtotal[2]);
//     printf("Avg:  %3.4f  %3.4f %3.4f\n", vaverage[0], vaverage[1], vaverage[2]);
//     printf("Min:  %3.4f  %3.4f %3.4f\n", vmin[0],     vmin[1],     vmin[2]);
//     printf("Max:  %3.4f  %3.4f %3.4f\n", vmax[0],     vmax[1],     vmax[2]);
// }
