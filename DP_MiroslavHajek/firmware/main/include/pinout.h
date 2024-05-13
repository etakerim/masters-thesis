#ifndef PINOUT_H
#define PINOUT_H

/**
 * @defgroup datalogger Accelerometer Data logger
 * @brief Data flow control from the sensor to the memory card
 *  @{
 */

#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "iis3dwb.h"
#include "hal/gpio_types.h"
#include "driver/sdmmc_host.h"
#include "driver/spi_master.h"
#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/FreeRTOS.h"

/**
 * @brief Maximum length of the file name buffer
 */
#define MAX_FILENAME            256
/**
 * @brief Directory in virtual file system (VFS) where microSD card gets mounted
 */
#define MOUNT_POINT             "/sd"
/**
 * @brief Path prefix of the directory where recordings are saved
 */
#define LOG_FOLDER              MOUNT_POINT"/"
/**
 * @brief Minimal possible delay in milliseconds for using a synchronization primitive
 */
#define NO_WAIT                 10 / portTICK_PERIOD_MS
/**
 * @brief Period in milliseconds for which the button is disabled after the press to prevent the bouncing effect
 */
#define SWITCH_DEBOUNCE         2000 / portTICK_PERIOD_MS

/**
 * @brief SD/MMC bus GPIO pin for CLK
 */
#define CARD_CLK_PIN            14
/**
 * @brief SD/MMC bus GPIO pin for CMD
 */
#define CARD_CMD_PIN            15
/**
 * @brief SD/MMC bus GPIO pin for D0
 */
#define CARD_D0_PIN             2

/**
 * @brief GPIO pin for a button that starts and stops recording
 */
#define RECORD_SWITCH_PIN       34
/**
 * @brief GPIO pin for the indicator LED
 */
#define RECORD_LED_PIN          32

/**
 * @brief GPIO pin for accelerometer SPI Master In Slave Out
 */
#define SENSOR_MISO             13
/**
 * @brief GPIO pin for accelerometer SPI Master Out Slave In
 */
#define SENSOR_MOSI             16
/**
 * @brief GPIO pin for accelerometer SPI Clock
 */
#define SENSOR_CLK              4
/**
 * @brief GPIO pin for accelerometer SPI Chip Select
 */
#define SENSOR_CS               5
/**
 * @brief GPIO pin for accelerometer interrupt pin
 */
#define SENSOR_INT1             33
/**
 * @brief SPI master bus frequency
 */
#define SPI_BUS_FREQUENCY       SPI_MASTER_FREQ_8M
/**
 * @brief Length of accelerometer FIFO buffer
 */
#define FIFO_LENGTH             512
/**
 * @brief Half length of accelerometer FIFO buffer
 */
#define FIFO_WATERMARK          FIFO_LENGTH / 2

/**
 * @brief Number of columns per acceleration vector
 */
#define NUM_OF_FIELDS           4
/**
 * @brief Length of buffer for SPI transaction
 */
#define SENSOR_SPI_LENGTH       NUM_OF_FIELDS * FIFO_LENGTH
/**
 * @brief Number of FIFO buffers that can be pushed to Queue before file write
 */
#define QUEUE_LENGTH            16


/**
 * @defgroup sensor Accelerometer
 * @brief Configuration of the accelerometer sensor
 *  @{
 */

/**
 * @brief Interval of the periodic timer in milliseconds that reads out circa half of accelerometer FIFO
 */
#define SAMPLE_RATE             9000
/**
 * @brief Hardware bus for accelerometer SPI interface
 */
#define SPI_BUS                 SPI3_HOST
/**
 * @brief Duration of recoding in microseconds (60 s)
 */
#define AUTO_TURN_OFF_US        60000000
/**
 * @brief Resolution of the accelerometer in a unit of g
 */
#define ACC_RESOLUTION          IIS3DWB_4g

/**
 * @brief Unprocessed data packet for storing samples from accelerometer
 */
typedef struct {
    uint16_t len;               /**< @brief  Number of samples in every array in the structure**/
    int32_t t[FIFO_LENGTH];     /**< @brief  Array of timestamps relative to time when sensor was enabled **/
    int32_t x[FIFO_LENGTH];     /**< @brief  Accelerometer samples for X axis **/
    int32_t y[FIFO_LENGTH];     /**< @brief  Accelerometer samples for Y axis **/
    int32_t z[FIFO_LENGTH];     /**< @brief  Accelerometer samples for Z axis **/
} Acceleration;

/**
 * @brief Configure SPI bus and set accelerometer to required parameters
 *
 * @param[in]   spi_dev   SPI bus
 * @param[in]   dev       Accelerometer sensor
 *
 * @return Status code of successful setup
 */
int sensor_enable(spi_device_handle_t *spi_dev, stmdev_ctx_t *dev);

/**
 * @brief Remove accelerometer from the SPI bus and disable it
 *
 * @param[in]   spi_dev     SPI bus
 */
void sensor_disable(spi_device_handle_t spi_dev);

/**
 * @brief Enable accelerometer interrupts
 *
 * @param[in]   dev     Accelerometer sensor
 */
void sensor_events_enable(stmdev_ctx_t *dev);

/**
 * @brief Disable accelerometer interrupts
 *
 * @param[in]   dev     Accelerometer sensor
 */
void sensor_events_disable(stmdev_ctx_t *dev);

/** @} */



/**
 * @brief Signal fatal error of system indiacted by blinking LED and halting execution
 *
 * @param[in]   delay   Interval in milliseconds for LED blink
 */
void panic(int delay);


/**
 * @defgroup storage Memory card
 * @brief Filesystem operations of the SD card
 *  @{
 */

/**
 * @brief Configure SD/MMC bus and mount SD memory card to FAT filesystem
 *
 * @param[in]   mount_point     path where the partition will be registered
 *
 * @return SD/MMC card information structure
 */
sdmmc_card_t *storage_enable(const char *mount_point);

/**
 * @brief Disable and unmount SD memory card from FAT filesystem
 *
 * @param[in]   card             SD/MMC card information structure
 * @param[in]   mount_point      path where partition is registered
 *
 * @return c
 */
void storage_disable(sdmmc_card_t *card, const char *mount_point);

/**
 * @brief Get file name for new recording with sequentially higher unused number
 *
 * @param[out]  filename   Available file name for new file
 * @param[in]   path       Base path prefix for saving the recording
 */
void get_recording_filename(char *filename, const char *path);

/** @} */


/**
 * @defgroup button Button
 * @brief Configuration of the recording button
 *  @{
 */

/**
 * @brief Configure GPIO input pin and interrupt handler for button press
 *
 * @param[in]   on              decides whether the button is enabled or disabled
 * @param[in]   isr_handler     handler function for button press in interrupt context
 */
void switch_enable(bool on, gpio_isr_t isr_handler);

/**
 * @brief Remove interrupt handler for button press
 */
void switch_disable(void);

/** @} */


/**
 * @defgroup led LED
 * @brief Configuration of the indicator LED light
 *  @{
 */

/**
 * @brief Configure GPIO for LED to output mode
 */
void led_enable(void);
/**
 * @brief Set the LED state
 *
 * @param[in]   on       turns LED light to be either on or off
 */
void led_light(bool on);

/** @} */


/** @} */

#endif
