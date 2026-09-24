#include "nanomt.hpp"
#include "ui.hpp"
#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

static const char* TAG="nanotranslate";
static Ui ui; static std::string input,output; static int kr=0,kc=0;
static const gpio_num_t JOYS[]={(gpio_num_t)CONFIG_NT_JOY_UP,(gpio_num_t)CONFIG_NT_JOY_DOWN,(gpio_num_t)CONFIG_NT_JOY_LEFT,(gpio_num_t)CONFIG_NT_JOY_RIGHT,(gpio_num_t)CONFIG_NT_JOY_PRESS};
static bool down(int i){return gpio_get_level(JOYS[i])==0;}
static void redraw(bool busy=false){ui.render(input,output,kr,kc,busy);ui.flush();}
static void pop_utf8(std::string&s){if(s.empty())return;size_t i=s.size()-1;while(i>0&&((unsigned char)s[i]&0xc0)==0x80)i--;s.erase(i);}

extern "C" void app_main(){
    ESP_LOGI(TAG,"NanoTranslate-S3 boot; internal=%u KB psram=%u KB",(unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)/1024),(unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1024));
    gpio_config_t g={};for(auto p:JOYS)g.pin_bit_mask|=1ULL<<p;g.mode=GPIO_MODE_INPUT;g.pull_up_en=GPIO_PULLUP_ENABLE;gpio_config(&g);
    if(!ui.init()){ESP_LOGE(TAG,"LCD init failed");return;}
    const esp_partition_t* part=esp_partition_find_first(ESP_PARTITION_TYPE_DATA,(esp_partition_subtype_t)0x40,"model");if(!part){ESP_LOGE(TAG,"model partition missing");return;}
    const void* mapped=nullptr;esp_partition_mmap_handle_t mh=0;if(esp_partition_mmap(part,0,part->size,ESP_PARTITION_MMAP_DATA,&mapped,&mh)!=ESP_OK){ESP_LOGE(TAG,"model mmap failed");return;}
    uint32_t total=0;std::memcpy(&total,(const uint8_t*)mapped+8+17*4,4);if(total<128||total>part->size){ESP_LOGE(TAG,"bad model length %u",(unsigned)total);return;}
    static nanomt::Model model;std::string err;if(!model.load(mapped,total,&err)){ESP_LOGE(TAG,"model: %s",err.c_str());return;}static std::unique_ptr<nanomt::Translator> tr;tr.reset(new nanomt::Translator(model));
    ESP_LOGI(TAG,"model loaded: vocab=%u d=%u enc=%u dec=%u",model.config().vocab,model.config().d_model,model.config().n_encoder,model.config().n_decoder);redraw();
    bool prev[5]={};int64_t press_ms=0;
    while(true){bool cur[5];for(int i=0;i<5;i++)cur[i]=down(i);bool changed=false;
        if(cur[0]&&!prev[0]){kr=(kr+5)%6;kc=std::min(kc,ui.cols(kr)-1);changed=true;}
        if(cur[1]&&!prev[1]){kr=(kr+1)%6;kc=std::min(kc,ui.cols(kr)-1);changed=true;}
        if(cur[2]&&!prev[2]){kc=(kc+ui.cols(kr)-1)%ui.cols(kr);changed=true;}
        if(cur[3]&&!prev[3]){kc=(kc+1)%ui.cols(kr);changed=true;}
        if(cur[4]&&!prev[4])press_ms=xTaskGetTickCount()*portTICK_PERIOD_MS;
        bool fire=false;if(!cur[4]&&prev[4]){const char*k=ui.key_at(kr,kc);if(k){if(std::strcmp(k,"BK")==0)pop_utf8(input);else if(std::strcmp(k,"SP")==0){if(input.size()<160)input+=' ';}else if(std::strcmp(k,"CLR")==0){input.clear();output.clear();}else if(std::strcmp(k,"GO")==0)fire=true;else if(input.size()<160)input+=k;}changed=true;}
        if(cur[4]&&prev[4]&&(xTaskGetTickCount()*portTICK_PERIOD_MS-press_ms)>900){fire=true;press_ms=INT64_MAX/2;}
        if(fire&&!input.empty()){redraw(true);nanomt::Stats st;std::string out;if(tr->translate(input,out,&st,&err)){output=out;ESP_LOGI(TAG,"src=%u out=%u enc=%.1fms dec=%.1fms total=%.1fms internal=%uKB psram=%uKB",st.src_tokens,st.out_tokens,st.encoder_us/1000.0,st.decoder_us/1000.0,st.total_us/1000.0,(unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)/1024),(unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1024));}else output="ERR: "+err;changed=true;}
        if(changed)redraw();for(int i=0;i<5;i++)prev[i]=cur[i];vTaskDelay(pdMS_TO_TICKS(35));
    }
}
