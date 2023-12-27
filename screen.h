#ifndef SCREENH
#define SCREENH

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

class Screen
{
public:
    Screen(uint32_t width, uint32_t height, float zoom_scale);

    void pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b);
    void pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a);

    void clear() { memset(pTextureBuffer, 0, width*height*sizeof(uint32_t)); };
    void setTitle(const char *title) { SDL_SetWindowTitle(mpWindow, title); };

    void quit(bool freeTextureBuffer);
    void show();
    void save(const char *file_name);

    uint32_t width;
    uint32_t height;
    uint32_t* pTextureBuffer;

private:
    SDL_Window* mpWindow;
    SDL_Renderer* mpRenderer;
    SDL_Texture* mpTexture;
};

Screen::Screen(uint32_t w, uint32_t h, float zoom_scale)
{
    width = w; height = h;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(width*zoom_scale, h*zoom_scale, 0, &mpWindow, &mpRenderer);
    SDL_RenderSetScale(mpRenderer, zoom_scale, zoom_scale);
    mpTexture = SDL_CreateTexture(mpRenderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, width, h);
    pTextureBuffer = new uint32_t[ w * h ];
}

#if __BYTE_ORDER == __LITTLE_ENDIAN
    void Screen::pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b)
    { pTextureBuffer[y*width + x] = (r<<24) | (g<<16) | (b<<8) | 0xFF; }
    void Screen::pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    { pTextureBuffer[y*width + x] = (r<<24) | (g<<16) | (b<<8) | a; }
#elif __BYTE_ORDER == __BIG_ENDIAN
    void Screen::pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b)
    { pTextureBuffer[y*width + x] = (0xFF<<24) | (b<<16) | (g<<8) | r; }
    void Screen::pixel(uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    { pTextureBuffer[y*width + x] = (a<<24) | (b<<16) | (g<<8) | r; }
#else
# error "Please fix <bits/endian.h>"
#endif


void Screen::show()
{
    SDL_UpdateTexture(mpTexture, NULL, pTextureBuffer, width * sizeof(uint32_t));
    SDL_RenderCopy(mpRenderer, mpTexture, NULL, NULL);
    SDL_RenderPresent(mpRenderer);
}

void Screen::save(const char *pFileName)
{
    SDL_Texture* target = SDL_GetRenderTarget(mpRenderer);
    SDL_SetRenderTarget(mpRenderer, mpTexture);
    int width, height;
    SDL_QueryTexture(mpTexture, NULL, NULL, &width, &height);
    SDL_Surface* surface = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);
    SDL_RenderReadPixels(mpRenderer, NULL, surface->format->format, surface->pixels, surface->pitch);
    IMG_SavePNG(surface, pFileName);
    SDL_FreeSurface(surface);
    SDL_SetRenderTarget(mpRenderer, target);
}

void Screen::quit(bool freeTextureBuffer)
{
    SDL_DestroyRenderer(mpRenderer);
    SDL_DestroyWindow(mpWindow);
    if (freeTextureBuffer) delete[] pTextureBuffer;
}


#endif
