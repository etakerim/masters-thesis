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
#define NO_WAIT                 10 / portTICK_PERIOD_MS

#define CARD_CLK_PIN            14
#define CARD_CMD_PIN            15
#define CARD_D0_PIN             2

#define RECORD_SWITCH_PIN       34
#define RECORD_LED_PIN          32

// UEXT connector
#define SENSOR_MISO             13
#define SENSOR_MOSI             16
#define SENSOR_CLK              4
#define SENSOR_CS               5
#define SENSOR_INT1             33
#define SPI_BUS_FREQUENCY       SPI_MASTER_FREQ_8M
#define FIFO_LENGTH             512
#define FIFO_WATERMARK          FIFO_LENGTH / 2

#define NUM_OF_FIELDS           4
#define SENSOR_SPI_LENGTH       NUM_OF_FIELDS * FIFO_LENGTH
#define QUEUE_LENGTH            16


// TIMER = 9 ms (1 sample = 1000 / 26667 = 0.0374 ms)
// Half full FIFO (256 samples = 9.6 ms)
#define SAMPLE_RATE             9000
#define SPI_BUS                 SPI3_HOST

typedef struct {
    float x[FIFO_LENGTH];
    float y[FIFO_LENGTH];
    float z[FIFO_LENGTH];
    int32_t t[FIFO_LENGTH];
    uint16_t len;
} Acceleration;



sdmmc_card_t *storage_enable(const char *mount_point);
void storage_disable(sdmmc_card_t *card, const char *mount_point);
void get_recording_filename(char *filename, const char *path);

void switch_enable(bool on, gpio_isr_t isr_handler);
void switch_disable(void);

void led_enable(void);
void led_light(bool on);

int sensor_enable(spi_device_handle_t *spi_dev, stmdev_ctx_t *dev);
void sensor_disable(spi_device_handle_t spi_dev);
void sensor_events_enable(stmdev_ctx_t *dev);
void sensor_events_disable(stmdev_ctx_t *dev);