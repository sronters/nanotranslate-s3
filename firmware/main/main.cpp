#include "nanomt.hpp"
#include "ui.hpp"

#include "driver/gpio.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

static const char* TAG = "nanotranslate";

static Ui g_ui;
static std::string g_input;
static std::string g_output;
static int g_keyboard_row = 0;
static int g_keyboard_col = 0;

static const gpio_num_t JOYSTICK_PINS[] = {
    static_cast<gpio_num_t>(CONFIG_NT_JOY_UP),
    static_cast<gpio_num_t>(CONFIG_NT_JOY_DOWN),
    static_cast<gpio_num_t>(CONFIG_NT_JOY_LEFT),
    static_cast<gpio_num_t>(CONFIG_NT_JOY_RIGHT),
    static_cast<gpio_num_t>(CONFIG_NT_JOY_PRESS),
};

static bool joystick_down(int index) {
    return gpio_get_level(JOYSTICK_PINS[index]) == 0;
}

static int64_t monotonic_ms() {
    return static_cast<int64_t>(xTaskGetTickCount()) *
           static_cast<int64_t>(portTICK_PERIOD_MS);
}

static void redraw(bool busy = false) {
    g_ui.render(g_input, g_output,
                g_keyboard_row, g_keyboard_col, busy);
    g_ui.flush();
}

static void pop_utf8(std::string& text) {
    if (text.empty()) return;

    size_t i = text.size() - 1;
    while (i > 0 &&
           (static_cast<unsigned char>(text[i]) & 0xc0U) == 0x80U) {
        --i;
    }
    text.erase(i);
}

static bool initialize_joystick() {
    gpio_config_t config = {};
    for (gpio_num_t pin : JOYSTICK_PINS) {
        config.pin_bit_mask |= 1ULL << static_cast<unsigned>(pin);
    }
    config.mode = GPIO_MODE_INPUT;
    config.pull_up_en = GPIO_PULLUP_ENABLE;
    config.pull_down_en = GPIO_PULLDOWN_DISABLE;
    config.intr_type = GPIO_INTR_DISABLE;
    return gpio_config(&config) == ESP_OK;
}

static bool load_model_partition(nanomt::Model& model,
                                 esp_partition_mmap_handle_t& map_handle,
                                 std::string& error) {
    const esp_partition_t* partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA,
        static_cast<esp_partition_subtype_t>(0x40),
        "model");
    if (!partition) {
        error = "model partition missing";
        return false;
    }

    const void* mapped = nullptr;
    if (esp_partition_mmap(
            partition,
            0,
            partition->size,
            ESP_PARTITION_MMAP_DATA,
            &mapped,
            &map_handle) != ESP_OK) {
        error = "model mmap failed";
        return false;
    }

    // .s3mt: 8-byte magic followed by 18 little-endian uint32 fields.
    // total_bytes is field h[17], therefore byte offset 8 + 17*4 = 76.
    constexpr size_t total_bytes_offset = 8 + 17 * sizeof(uint32_t);
    if (partition->size < total_bytes_offset + sizeof(uint32_t)) {
        error = "model partition too small";
        return false;
    }

    uint32_t total_bytes = 0;
    std::memcpy(
        &total_bytes,
        static_cast<const uint8_t*>(mapped) + total_bytes_offset,
        sizeof(total_bytes));

    if (total_bytes < 128U || total_bytes > partition->size) {
        error = "bad model length";
        return false;
    }

    if (!model.load(mapped, total_bytes, &error)) {
        return false;
    }
    return true;
}

extern "C" void app_main() {
    ESP_LOGI(
        TAG,
        "NanoTranslate-S3 boot; internal=%u KB psram=%u KB",
        static_cast<unsigned>(
            heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
        static_cast<unsigned>(
            heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));

    if (!initialize_joystick()) {
        ESP_LOGE(TAG, "joystick GPIO init failed");
        return;
    }

    if (!g_ui.init()) {
        ESP_LOGE(TAG, "LCD init failed");
        return;
    }

    static nanomt::Model model;
    static esp_partition_mmap_handle_t model_map_handle = 0;
    std::string error;
    if (!load_model_partition(model, model_map_handle, error)) {
        ESP_LOGE(TAG, "model load failed: %s", error.c_str());
        return;
    }

    static std::unique_ptr<nanomt::Translator> translator =
        std::make_unique<nanomt::Translator>(model);

    ESP_LOGI(
        TAG,
        "model loaded: vocab=%u d=%u enc=%u dec=%u",
        model.config().vocab,
        model.config().d_model,
        model.config().n_encoder,
        model.config().n_decoder);

    redraw();

    bool previous[5] = {};
    int64_t press_started_ms = 0;
    bool long_press_fired = false;

    while (true) {
        bool current[5] = {};
        for (int i = 0; i < 5; ++i) {
            current[i] = joystick_down(i);
        }

        bool changed = false;

        if (current[0] && !previous[0]) {
            g_keyboard_row = (g_keyboard_row + g_ui.rows() - 1) %
                             g_ui.rows();
            g_keyboard_col = std::min(
                g_keyboard_col, g_ui.cols(g_keyboard_row) - 1);
            changed = true;
        }
        if (current[1] && !previous[1]) {
            g_keyboard_row = (g_keyboard_row + 1) % g_ui.rows();
            g_keyboard_col = std::min(
                g_keyboard_col, g_ui.cols(g_keyboard_row) - 1);
            changed = true;
        }
        if (current[2] && !previous[2]) {
            const int columns = g_ui.cols(g_keyboard_row);
            g_keyboard_col =
                (g_keyboard_col + columns - 1) % columns;
            changed = true;
        }
        if (current[3] && !previous[3]) {
            const int columns = g_ui.cols(g_keyboard_row);
            g_keyboard_col = (g_keyboard_col + 1) % columns;
            changed = true;
        }

        if (current[4] && !previous[4]) {
            press_started_ms = monotonic_ms();
            long_press_fired = false;
        }

        bool translate_now = false;
        if (!current[4] && previous[4] && !long_press_fired) {
            const char* key =
                g_ui.key_at(g_keyboard_row, g_keyboard_col);
            if (key) {
                if (std::strcmp(key, "BK") == 0) {
                    pop_utf8(g_input);
                } else if (std::strcmp(key, "SP") == 0) {
                    if (g_input.size() < 160) g_input.push_back(' ');
                } else if (std::strcmp(key, "CLR") == 0) {
                    g_input.clear();
                    g_output.clear();
                } else if (std::strcmp(key, "GO") == 0) {
                    translate_now = true;
                } else if (g_input.size() < 160) {
                    g_input += key;
                }
            }
            changed = true;
        }

        if (current[4] && previous[4] && !long_press_fired &&
            monotonic_ms() - press_started_ms > 900) {
            translate_now = true;
            long_press_fired = true;
        }

        if (translate_now && !g_input.empty()) {
            redraw(true);

            nanomt::Stats stats;
            std::string translated;
            if (translator->translate(
                    g_input, translated, &stats, &error)) {
                g_output = translated;
                ESP_LOGI(
                    TAG,
                    "src=%u out=%u enc=%.1fms dec=%.1fms "
                    "total=%.1fms internal=%uKB psram=%uKB",
                    stats.src_tokens,
                    stats.out_tokens,
                    stats.encoder_us / 1000.0,
                    stats.decoder_us / 1000.0,
                    stats.total_us / 1000.0,
                    static_cast<unsigned>(
                        heap_caps_get_free_size(MALLOC_CAP_INTERNAL) /
                        1024),
                    static_cast<unsigned>(
                        heap_caps_get_free_size(MALLOC_CAP_SPIRAM) /
                        1024));
            } else {
                g_output = "ERR: " + error;
            }
            changed = true;
        }

        if (changed) {
            redraw();
        }

        for (int i = 0; i < 5; ++i) {
            previous[i] = current[i];
        }

        vTaskDelay(pdMS_TO_TICKS(35));
    }
}
