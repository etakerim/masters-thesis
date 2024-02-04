#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "iis3dwb.h"
#include "hal/gpio_types.h"
#include "driver/sdmmc_host.h"
#include "driver/spi_master.h"
#include "esp_err.h"
#include "esp_log.h"


#define MAX_FILENAME            256
#define MOUNT_POINT             "/sd"
#define LOG_FOLDER              MOUNT_POINT"/"
//MOUNT_POINT"/accelerometer"

#define CARD_CLK_PIN            14
#define CARD_CMD_PIN            15
#define CARD_D0_PIN             2

#define RECORD_SWITCH_PIN       34
#define RECORD_LED_PIN          3

// UEXT connector
#define SENSOR_MISO             13
#define SENSOR_MOSI             16
#define SENSOR_CLK              4
#define SENSOR_CS               5
#define SENSOR_INT1             36
#define SPI_BUS_FREQUENCY       SPI_MASTER_FREQ_8M
#define FIFO_WATERMARK          256
#define SPI_BUS                 SPI3_HOST


sdmmc_card_t *storage_enable(const char *mount_point);
void storage_disable(sdmmc_card_t *card, const char *mount_point);
void get_recording_filename(char *filename, const char *path);

void switch_enable(bool on, gpio_isr_t isr_handler);
void switch_disable(void);

void led_enable(void);
void led_light(bool on);

void sensor_enable(spi_device_handle_t *spi_dev, stmdev_ctx_t *dev);
void sensor_disable(spi_device_handle_t spi_dev);
void sensor_int_threshold_enable(stmdev_ctx_t *dev, gpio_isr_t isr_handler);
void sensor_int_threshold_disable(void);
void sensor_read(stmdev_ctx_t *dev, FILE *output);