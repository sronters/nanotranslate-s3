#include "ui.hpp"
#include "font8x12.h"
#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_heap_caps.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <algorithm>
#include <cstring>

static const char* TAG="ui";
static SemaphoreHandle_t g_flush_done=nullptr;
static bool on_color_done(esp_lcd_panel_io_handle_t, esp_lcd_panel_io_event_data_t*, void*){ BaseType_t hp=pdFALSE; if(g_flush_done)xSemaphoreGiveFromISR(g_flush_done,&hp); return hp==pdTRUE; }
static const char* K0[]={"й","ц","у","к","е","н","г"};
static const char* K1[]={"ш","щ","з","х","ъ","ф","ы"};
static const char* K2[]={"в","а","п","р","о","л","д"};
static const char* K3[]={"ж","э","я","ч","с","м","и"};
static const char* K4[]={"т","ь","б","ю","ё",",","."};
static const char* K5[]={"BK","SP","?","!","CLR","GO"};
static const char** KS[]={K0,K1,K2,K3,K4,K5}; static const int KN[]={7,7,7,7,7,6};
static uint16_t rgb565(uint8_t r,uint8_t g,uint8_t b){return ((r&0xf8)<<8)|((g&0xfc)<<3)|(b>>3);} 
static uint32_t utf8next(const char*& p){const unsigned char*s=(const unsigned char*)p;if(!*s)return 0;if(*s<0x80){p++;return *s;}if((*s&0xe0)==0xc0){uint32_t c=((*s&0x1f)<<6)|(s[1]&0x3f);p+=2;return c;}if((*s&0xf0)==0xe0){uint32_t c=((*s&0x0f)<<12)|((s[1]&0x3f)<<6)|(s[2]&0x3f);p+=3;return c;}p++;return '?';}
static const uint8_t* glyph(uint32_t cp){for(size_t i=0;i<NT_FONT_COUNT;i++)if(NT_FONT[i].cp==cp)return NT_FONT[i].row;for(size_t i=0;i<NT_FONT_COUNT;i++)if(NT_FONT[i].cp=='?')return NT_FONT[i].row;return nullptr;}

bool Ui::init(){
    spi_bus_config_t bus={}; bus.sclk_io_num=CONFIG_NT_LCD_SCLK;bus.mosi_io_num=CONFIG_NT_LCD_MOSI;bus.miso_io_num=-1;bus.quadwp_io_num=-1;bus.quadhd_io_num=-1;bus.max_transfer_sz=240*20*2;
    if(spi_bus_initialize(SPI2_HOST,&bus,SPI_DMA_CH_AUTO)!=ESP_OK)return false;
    esp_lcd_panel_io_spi_config_t io={};io.cs_gpio_num=CONFIG_NT_LCD_CS;io.dc_gpio_num=CONFIG_NT_LCD_DC;io.spi_mode=0;io.pclk_hz=40*1000*1000;io.trans_queue_depth=10;io.lcd_cmd_bits=8;io.lcd_param_bits=8;
    esp_lcd_panel_io_handle_t ioh=nullptr;if(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)SPI2_HOST,&io,&ioh)!=ESP_OK)return false;
    g_flush_done=xSemaphoreCreateBinary(); if(!g_flush_done)return false; esp_lcd_panel_io_callbacks_t cbs={}; cbs.on_color_trans_done=on_color_done; ESP_ERROR_CHECK(esp_lcd_panel_io_register_event_callbacks(ioh,&cbs,nullptr));
    esp_lcd_panel_dev_config_t pc={};pc.reset_gpio_num=CONFIG_NT_LCD_RST;pc.rgb_ele_order=LCD_RGB_ELEMENT_ORDER_RGB;pc.bits_per_pixel=16;
    if(esp_lcd_new_panel_st7789(ioh,&pc,&panel_)!=ESP_OK)return false;esp_lcd_panel_reset(panel_);esp_lcd_panel_init(panel_);esp_lcd_panel_set_gap(panel_,CONFIG_NT_LCD_X_GAP,CONFIG_NT_LCD_Y_GAP);esp_lcd_panel_disp_on_off(panel_,true);
    gpio_config_t bl={};bl.pin_bit_mask=1ULL<<CONFIG_NT_LCD_BL;bl.mode=GPIO_MODE_OUTPUT;gpio_config(&bl);gpio_set_level((gpio_num_t)CONFIG_NT_LCD_BL,1);
    fb_=(uint16_t*)heap_caps_malloc(240*240*2,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT);
    strip_=(uint16_t*)heap_caps_malloc(240*20*2,MALLOC_CAP_INTERNAL|MALLOC_CAP_DMA);
    if(!fb_||!strip_){ESP_LOGE(TAG,"framebuffer/strip alloc failed");return false;}return true;
}
void Ui::clear(uint16_t c){std::fill(fb_,fb_+240*240,c);}void Ui::rect(int x,int y,int w,int h,uint16_t c){int x1=std::max(0,x),y1=std::max(0,y),x2=std::min(240,x+w),y2=std::min(240,y+h);for(int yy=y1;yy<y2;yy++)for(int xx=x1;xx<x2;xx++)fb_[yy*240+xx]=c;}
void Ui::text(int x,int y,const std::string&s,uint16_t c,int max_chars){const char*p=s.c_str();int n=0;while(*p&&(max_chars<0||n<max_chars)){uint32_t cp=utf8next(p);const uint8_t*g=glyph(cp);if(g){for(int yy=0;yy<12;yy++)for(int xx=0;xx<8;xx++)if(g[yy]&(1<<(7-xx))){int px=x+xx,py=y+yy;if(px>=0&&px<240&&py>=0&&py<240)fb_[py*240+px]=c;}}x+=8;n++;if(x>232)break;}}
int Ui::cols(int r)const{return r>=0&&r<6?KN[r]:0;}const char* Ui::key_at(int r,int c)const{return(r>=0&&r<6&&c>=0&&c<KN[r])?KS[r][c]:nullptr;}
void Ui::render(const std::string& in,const std::string& out,int sr,int sc,bool busy){uint16_t bg=rgb565(10,12,16),fg=rgb565(235,238,245),muted=rgb565(120,130,145),hi=rgb565(35,95,190),key=rgb565(28,32,42);clear(bg);text(4,3,"RU >",muted);text(4,17,in,fg,29);text(4,34,"EN >",muted);text(4,48,busy?"translating...":out,fg,29);int top=82;
    for(int r=0;r<6;r++){int n=KN[r],kw=236/n;for(int c=0;c<n;c++){int x=2+c*kw,y=top+r*25;rect(x,y,kw-2,22,(r==sr&&c==sc)?hi:key);std::string label=KS[r][c];int glyphs=((unsigned char)label[0]>=0x80)?1:(int)label.size();int tx=x+std::max(2,(kw-8*std::min(glyphs,3))/2);text(tx,y+5,label,fg,3);}}
}
void Ui::flush(){if(!panel_||!fb_||!strip_)return; for(int y=0;y<240;y+=20){int h=std::min(20,240-y);std::memcpy(strip_,fb_+y*240,size_t(h)*240*2); if(esp_lcd_panel_draw_bitmap(panel_,0,y,240,y+h,strip_)!=ESP_OK)return; xSemaphoreTake(g_flush_done,portMAX_DELAY);}}
