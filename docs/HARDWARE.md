# Hardware and wiring

Target module: **ESP32-S3-WROOM-1 N16R8** (16 MB flash, 8 MB octal PSRAM).

The firmware defaults to a generic **240x240 ST7789 SPI display** and a five-way digital joystick. All pins are configurable under `idf.py menuconfig -> NanoTranslate hardware`.

Default example wiring:

| Signal | GPIO |
|---|---:|
| LCD MOSI | 11 |
| LCD SCLK | 12 |
| LCD CS | 10 |
| LCD DC | 9 |
| LCD RST | 8 |
| LCD BL | 7 |
| joystick up | 4 |
| joystick down | 5 |
| joystick left | 6 |
| joystick right | 13 |
| joystick press | 14 |

The joystick inputs are configured with internal pull-ups, so each switch should short the input to **GND** when pressed. If your joystick module has active electronics rather than five dry contacts, verify its voltage/interface before connecting it.

## Display notes

ST7789 modules differ in visible window offsets. If the image is shifted, adjust `NT_LCD_X_GAP` and `NT_LCD_Y_GAP` in menuconfig. The framebuffer is 240x240 RGB565 (~112.5 KiB) in PSRAM and is flushed over SPI at 40 MHz.

## N16R8 pin warning

Do not blindly copy pinouts from an ESP32-S3 module with a different flash/PSRAM configuration. The N16R8 package uses octal memory internally; use GPIOs actually broken out by your board and keep the display/joystick pins configurable.

## Model partition

`firmware/partitions.csv` reserves 2 MiB at `0x210000` for `nanotranslate.s3mt`. The default base model is designed to be around 1 MiB in Q8 form, leaving substantial headroom for experiments.
