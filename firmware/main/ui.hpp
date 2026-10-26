#pragma once
#include <cstdint>
#include <string>
#include "esp_lcd_panel_ops.h"

class Ui {
public:
    bool init();
    void render(const std::string& input,const std::string& output,int row,int col,bool busy=false);
    void flush();
    const char* key_at(int row,int col) const;
    int rows() const { return 6; }
    int cols(int row) const;
private:
    esp_lcd_panel_handle_t panel_=nullptr;
    uint16_t* fb_=nullptr;
    uint16_t* strip_=nullptr;
    void clear(uint16_t c); void rect(int x,int y,int w,int h,uint16_t c); void text(int x,int y,const std::string&s,uint16_t c,int max_chars=-1);
};
