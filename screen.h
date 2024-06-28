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

#endif
