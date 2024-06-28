#include <time.h>
#include <SDL2/SDL.h>
#include "hip-commons.h"

#include "macros.h"

#include "screen.hpp"
#include "vector.h"
#include "ray.h"
#include "camera.h"
#include "hitable.h"
#include "hitable_list.h"
#include "sphere.h"
#include "material.h"

#define checkHipErrors(val) checkHip( (val), #val, __FILE__, __LINE__ )
void checkHip(hipError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "HIP error " << static_cast<unsigned int>(result) << " " << hipGetErrorString(result)
                  << " at " << file << ":" << line << " '" << func << "'\n";
        hipDeviceReset();
        exit(static_cast<unsigned int>(result));
    }
}

// Kernels for memory management on the device. 'd' prefix represents
// device-only data.
#define NUM_WORLD_ELEMENTS 9
__global__ void allocWorld(Hitable **dList, Hitable **dWorld, Camera **dCamera)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) // only run once
    {
        dList[0] = new Sphere(vec3( 0,   100.6, -2  ), 100, new Diffuse(vec3(0.3, 0.5, 0.7)));
        dList[1] = new Sphere(vec3( 0,     0,   -2  ), 0.5, new Diffuse(vec3(0.8, 0.3, 0.3)));
        dList[2] = new Sphere(vec3( 2.6,  -1.4, -1.7), 0.7, new Diffuse(vec3(0.9, 0.5, 0.2)));
        dList[3] = new Sphere(vec3( 1,     0,   -2  ), 0.4, new Diffuse(vec3(0.3, 0.4, 0.9)));
        dList[4] = new Sphere(vec3(-0.3,   0.1, -1  ), 0.3, new Diffuse(vec3(0.5, 1.0, 0.6)));
        dList[5] = new Sphere(vec3( 0,     0.2,  1  ), 0.3, new Diffuse(vec3(0.8, 0.2, 0.3)));
        dList[6] = new Sphere(vec3(-1,    -0.3, -1.2), 0.2, new Diffuse(vec3(0.3, 0.2, 0.0)));
        dList[7] = new Sphere(vec3( 0.3,  -0.5, -1.1), 0.2, new Diffuse(vec3(0.0, 0.1, 0.9)));
        dList[8] = new Sphere(vec3( 0,    -5.0, -4  ), 2.0, new Diffuse(vec3(1.0, 1.0, 1.0)));
        *(dWorld)  = new HitableList(dList, NUM_WORLD_ELEMENTS);
        *(dCamera) = new Camera(
            vec3(0,0,2),
            vec3(0,0,-1),
            vec3(0,1,0),
            80,
            (float)WINDOW_WIDTH/WINDOW_HEIGHT
        );
    }
}

__global__ void freeWorld(Hitable **dList, Hitable **dWorld, Camera **dCamera)
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

__global__ void updateWorld(Camera **dCamera, float dx, float dy, float dz)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) // only run once
    {
        (*dCamera)->move(dx, dy, dz);
    }
}

// Sets up a random number state for each thread. This cannot be a global RNG,
// because each thread will call it simultaneously, and all recieve the same
// number.
__global__ void initRands(hiprandState *pRandStates)
{
    float i = threadIdx.x + blockIdx.x * blockDim.x;
    float j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= WINDOW_WIDTH) || (j >= WINDOW_HEIGHT)) return;
    uint32_t pixel_index = (uint32_t)j*WINDOW_WIDTH + (uint32_t)i;
    //Each thread gets same seed, a different sequence number, no offset
    hiprand_init(0xE621, pixel_index, 0, &pRandStates[pixel_index]);
}

__device__ vec3 color(ray *r, Hitable **world, hiprandState *pRandState)
{
    vec3 runningAttenuation = vec3(1,1,1);
    hit_record rec;
    vec3 attenuation;
    // do MAX_NUM_REFLECTIONS reflections
    for (int i = 0; i < MAX_NUM_REFLECTIONS; i++)
    {
        bool isLightSource = false;
        if ((*world)->hit(r, 0.001f, FLT_MAX, rec))
        {
            if (rec.pMat->scatter(r, rec, attenuation, pRandState, isLightSource))
                runningAttenuation *= attenuation;
            else return runningAttenuation * attenuation * isLightSource;
        }
        else
        {
            // return runningAttenuation * SKYBOX_COLOR;
            // lerp gray...blue and multiply by attenuation
            vec3 unit_direction = unit_vector(r->direction());
            float t = 0.5f*(unit_direction.y()+1.0f);
            return runningAttenuation * ((1.0f-t)*vec3(0.6, 0.8, 1.0) + t*vec3(0.4, 0.5, 0.6));
        }
    }

    // exceeded recursion
    return vec3(0,0,0);
}

// Render kernel -- steps through raytracing a pixel, NUM_ALIAS_STEPS times,
// and averages their values. This averaging step (1) smoothes out render
// artifacts, and (2) achieves an antialiasing effect.
__global__ void render(uint32_t *pFrameBuffer, Camera **pCam, Hitable **world, hiprandState *pRandStates)
{
    // __fmaf_rn(float a, float b, float c) == a*b+c
    const float i = __fmaf_rn(blockIdx.x, blockDim.x, threadIdx.x);
    const float j = __fmaf_rn(blockIdx.y, blockDim.y, threadIdx.y);
    if ((i >= WINDOW_WIDTH) || (j >= WINDOW_HEIGHT)) return;
    uint32_t pixel_index = (uint32_t)j*WINDOW_WIDTH + (uint32_t)i;
    hiprandState pLocalRandState = pRandStates[pixel_index];

    vec3 col(0,0,0);
    for (int iter = 0; iter < NUM_ALIAS_STEPS; iter++)
    {
        // add drand48() for a slight randomization to the direction.
        // this non-uniformity is what achieves the above benefits.
        // note: for CUDA, drand48() is not available. instead,
        // hiprand_uniform(hiprandState*) is equivalent.
        float u = (i /*+ hiprand_uniform(&pLocalRandState)*/) * (1.0f / WINDOW_WIDTH);
        float v = (j /*+ hiprand_uniform(&pLocalRandState)*/) * (1.0f / WINDOW_HEIGHT);

        ray r = (*pCam)->getRay(u, v);

        col += color(&r, world, &pLocalRandState);
    }
    // #define DUFF_DEVICE_16(nTimes, aAction) \
    // do { \
    // int times = (nTimes + 15) >> 4; \
    // switch (nTimes & 15) { \
    // case 0: do { aAction; \
    // case 15:     aAction; \
    // case 14:     aAction; \
    // case 13:     aAction; \
    // case 12:     aAction; \
    // case 11:     aAction; \
    // case 10:     aAction; \
    // case 9:      aAction; \
    // case 8:      aAction; \
    // case 7:      aAction; \
    // case 6:      aAction; \
    // case 5:      aAction; \
    // case 4:      aAction; \
    // case 3:      aAction; \
    // case 2:      aAction; \
    // case 1:      aAction; } while (--times > 0); \
    // } } while (0)
    //
    // float u;
    // float v;
    // ray r;
    //
    // DUFF_DEVICE_16(NUM_ALIAS_STEPS,
    //     u = (i + hiprand_uniform(&pLocalRandState)) * (1.0f / WINDOW_WIDTH);
    //     v = (j + hiprand_uniform(&pLocalRandState)) * (1.0f / WINDOW_HEIGHT);
    //     r = (*pCam)->getRay(u, v);
    //     col += color(&r, world, &pLocalRandState);
    // );

    col *= (1.0f / NUM_ALIAS_STEPS);

    // A square root is present because SDL assumes the image is gamma-
    // corrected. It is not. This is corrected by raising the color to the
    // power of 1/gamma. To simplify this math, gamma=2 is used, such that the
    // un-gamma-correct-ing is simply the square root of the color. The CUDA
    // __saturatef function is an efficient single-instruction call to clamp a
    // float from 0...1; note, x86 does not have an instruction like this.
    // Clamping the colors to 0-1 is important because light sources can go
    // above that, and cause overflow issues and really strange visual
    // glitches. The rest of this mess scales 0..1 --> 0..255, and converts to
    // uint8_t to chain together.
    uint8_t buffer[4];
    #if __BYTE_ORDER == __LITTLE_ENDIAN
        buffer[3] = uint8_t(__saturatef( sqrt(col[0]) ) * 255);
        buffer[2] = uint8_t(__saturatef( sqrt(col[1]) ) * 255);
        buffer[1] = uint8_t(__saturatef( sqrt(col[2]) ) * 255);
        buffer[0] = 0xFF;
    #elif __BYTE_ORDER == __BIG_ENDIAN
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
    // clock_t setup_start, render_start, teardown_start, teardown_stop;
    // setup_start = clock();
    // Thread and block setup
    // Larger blocks may run faster, since more threads can be used; however,
    // too large a block size will mean that cores will be doing dissimilar
    // operations, greatly reducing performance. GPUs are built and refined
    // for multiple similar operations at the same time, not multiple diverse
    // operations, per thread.
    dim3 blockDims(32,32);
    dim3 gridDims(float(WINDOW_WIDTH) / blockDims.x + 1, float(WINDOW_HEIGHT) / blockDims.y + 1);
    int num_pixels = WINDOW_WIDTH * WINDOW_HEIGHT;

    // Allocate framebuffer and send to cuda
    size_t fbSize = num_pixels*sizeof(uint32_t);
    uint32_t *pFrameBuffer;
    checkHipErrors(hipMallocManaged((void**)&pFrameBuffer, fbSize));
    // Allocate random state storage
    hiprandState *dRandStates;
    checkHipErrors(hipMalloc((void **)&dRandStates, num_pixels*sizeof(hiprandState)));
    // Allocate and set up world objects
    Hitable **dList;
    Hitable **dWorld;
    checkHipErrors(hipMalloc((void **)&dList,  sizeof(Hitable *) * NUM_WORLD_ELEMENTS));
    checkHipErrors(hipMalloc((void **)&dWorld, sizeof(Hitable *)));
    // Allocate camera
    Camera **dCamera;
    checkHipErrors(hipMalloc((void **)&dCamera, sizeof(Camera *)));

    // Device-side allocate world
    allocWorld<<<1,1>>>(dList, dWorld, dCamera);
    checkHipErrors(hipGetLastError());
    checkHipErrors(hipDeviceSynchronize());
    // Initialize RNG and per-thread state
    initRands<<<gridDims, blockDims>>>(dRandStates);
    checkHipErrors(hipGetLastError());
    checkHipErrors(hipDeviceSynchronize());

    // Set up SDL2 stuffs
    SDL_Event e;
    Screen screen(WINDOW_WIDTH, WINDOW_HEIGHT, 1);
    const uint8_t* keyb = SDL_GetKeyboardState(NULL);
    // change framebuffer to &fb, to speed up render time
    free(screen.pTextureBuffer);
    screen.pTextureBuffer = pFrameBuffer;
    screen.show();

    bool shouldCont = true;
    while (shouldCont)
    {
        // Render
        //SDL_Delay(10);
        // render_start = clock();
        render<<<gridDims, blockDims>>>(pFrameBuffer, dCamera, dWorld, dRandStates);
        checkHipErrors(hipGetLastError());
        checkHipErrors(hipDeviceSynchronize());
        // teardown_start = clock();
        screen.show();

        // Events
        while (SDL_PollEvent(&e))
        {
            switch (e.type)
            {
                case SDL_QUIT:
                    shouldCont = false;
                    break;
            }
        }

        // Update
        float dx = MOVE_SPEED * (keyb[SDL_SCANCODE_D] - keyb[SDL_SCANCODE_A]);
        float dy = MOVE_SPEED * (keyb[SDL_SCANCODE_Q] - keyb[SDL_SCANCODE_E]);
        float dz = MOVE_SPEED * (keyb[SDL_SCANCODE_S] - keyb[SDL_SCANCODE_W]);
        updateWorld<<<1,1>>>(dCamera, dx, dy, dz);
        checkHipErrors(hipGetLastError());
        checkHipErrors(hipDeviceSynchronize());
    }

    // teardown_stop = clock();
    // float setup_seconds =    ((float)(render_start - setup_start))     / CLOCKS_PER_SEC;
    // float render_seconds =   ((float)(teardown_start - render_start))  / CLOCKS_PER_SEC;
    // float teardown_seconds = ((float)(teardown_stop - teardown_start)) / CLOCKS_PER_SEC;
    // // return vec3(setup_seconds, render_seconds, teardown_seconds);
    // printf("Render complete.\n");
    // printf("Setup took:    %.5f seconds.\n", setup_seconds);
    // printf("Render took:   %.5f seconds.\n", render_seconds);
    // printf("Teardown took: %.5f seconds.\n", teardown_seconds);

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

    // screen.save("render.png");

    // Free objects, gpu side
    freeWorld<<<1,1>>>(dList, dWorld, dCamera);
    checkHipErrors(hipGetLastError());
    // Free object pointers, gpu side
    checkHipErrors(hipFree(dCamera));
    checkHipErrors(hipFree(dWorld));
    checkHipErrors(hipFree(dList));
    checkHipErrors(hipFree(dRandStates));
    checkHipErrors(hipFree(pFrameBuffer));

    screen.quit(false);
    SDL_Quit();
    return 0;
}
