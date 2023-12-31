#include <stdio.h>
#include <string.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"

#include "hal/gpio_types.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "iis3dwb_reg.h"


#define MAX_FILENAME            256
#define MOUNT_POINT             "/sd"
#define LOG_FOLDER              MOUNT_POINT"/acc"

#define CARD_CLK_PIN    14
#define CARD_CMD_PIN    15
#define CARD_D0_PIN     2


// Max fifo watermark is 511
#define SENSOR_MISO         16
#define SENSOR_MOSI         32
#define SENSOR_CLK          33
#define SENSOR_CS           13
#define SENSOR_INT1         35
#define SPI_BUS_FREQUENCY   SPI_MASTER_FREQ_8M
#define FIFO_WATERMARK      256
static spi_host_device_t    SPI_BUS = SPI2_HOST;
static iis3dwb_fifo_out_raw_t fifo_data[FIFO_WATERMARK];


sdmmc_card_t *storage_enable(const char *mount_point)
{
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = true,     // Set to false
        .max_files = 5,                     // Maximum number of opened files
        .allocation_unit_size = 16 * 1024   // Useful only for format
    };

    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 1;
    slot_config.clk = CARD_CLK_PIN;
    slot_config.cmd = CARD_CMD_PIN;
    slot_config.d0 = CARD_D0_PIN;

    sdmmc_card_t *card;
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    esp_err_t ret = esp_vfs_fat_sdmmc_mount(mount_point, &host, &slot_config, &mount_config, &card);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            // Failed to mount filesystem
        } else {
            // Failed to initialize the card
        }
        return NULL;
    }

    // Card has been initialized, print its properties
    // sdmmc_card_print_info(stdout, card);

    // Format FATFS
    // ret = esp_vfs_fat_sdcard_format(mount_point, card);

    return card;
}

void storage_disable(sdmmc_card_t *card, const char *mount_point) 
{
    // All done, unmount partition and disable SDMMC peripheral
    esp_vfs_fat_sdcard_unmount(mount_point, card);
}

unsigned long get_new_recording_name(const char *path) 
{
    unsigned long seq = 1;

    DIR *folder = opendir(path);
    if (dp == NULL)
        return seq;
    
    struct dirent *entry;
    while ((entry = readdir(folder)) != NULL) {
        // TODO
        // entry->d_name
        // Split by "." - write \0 in copy
        int name = atoi(parsed_name);
        if (name > seq)
            seq = name;
    }
          
    closedir(folder);
    return seq;
}



/////////////////////////////////////////////////////////////////////////

static int32_t platform_write(void *handle, uint8_t reg, const uint8_t *bufp, uint16_t len)
{
    spi_device_handle_t spi = *(spi_device_handle_t *)handle;
    spi_transaction_t t = {
        .addr = reg,
        .length = 8 * len,
        .tx_buffer = buffer
    };
    spi_device_transmit(spi, &t);
}

static int32_t platform_read(void *handle, uint8_t reg, uint8_t *bufp, uint16_t len);
{
    uint8_t tx_buffer[1] = {reg | 0x80};

    spi_transaction_t t = {
        .length = 8 * sizeof(tx_buffer),
        .rxlength = 8 * len,
        .tx_buffer = tx_buffer,
        .rx_buffer = bufp
    };
    spi_device_transmit(spi, &t);
}


static TaskHandle_t sample_tick;

static bool IRAM_ATTR isr_sample(void *args)
{
    BaseType_t higher_priority_woken = pdFALSE;
    vTaskNotifyGiveFromISR(sample_tick, &higher_priority_woken);
    return higher_priority_woken == pdTRUE;
}

void sampler_task()
{
    while (1) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            sensor_read();
        }
    }
}


void sensor_enable(void)
{
    esp_err_t err;

    // SPI bus
    spi_bus_config_t spi_bus = {
        .miso_io_num = SENSOR_MISO,
        .mosi_io_num = SENSOR_MOSI,
        .sclk_io_num = SENSOR_CLK,
        .max_transfer_sz = 1024
    };
    err = spi_bus_initialize(SPI_BUS, &spi_bus, SPI_DMA_DISABLED);  // TODO: SPI_DMA_CH_AUTO
    spi_device_interface_config_t spi_iface = {
        .clock_speed_hz=SPI_BUS_FREQUENCY,
        .flags=SPI_DEVICE_HALFDUPLEX,
        .address_bits=7,
        .mode=0,
        .spics_io_num=SENSOR_CS,
        .queue_size=5
    };
    spi_device_handle_t spi_dev;
    err = spi_bus_add_device(SPI_BUS, &spi_iface, &spi_dev);


    // Peripheral
    stmdev_ctx_t dev;
    dev.write_reg = platform_write;
    dev.read_reg = platform_read;
    dev.handle = &SPI_BUS;

    uint8_t who_am_i;
    iis3dwb_device_id_get(&dev, &who_am_i);
    if (who_am_i != IIS3DWB_ID)
        while (1);

    iis3dwb_reset_set(&dev, PROPERTY_ENABLE);

    uint8_t rst;
    do {
        iis3dwb_reset_get(&dev_ctx, &rst);
    } while (rst);

    iis3dwb_block_data_update_set(&dev, PROPERTY_ENABLE);

    iis3dwb_xl_full_scale_set(&dev, IIS3DWB_2g);
    iis3dwb_fifo_watermark_set(&dev, FIFO_WATERMARK);
    iis3dwb_fifo_xl_batch_set(&dev, IIS3DWB_XL_BATCHED_AT_26k7Hz);
    iis3dwb_fifo_mode_set(&dev, IIS3DWB_STREAM_MODE);

    /* Set Output Data Rate */
    iis3dwb_xl_data_rate_set(&dev, IIS3DWB_XL_ODR_26k7Hz);
    iis3dwb_fifo_timestamp_batch_set(&dev_ctx, IIS3DWB_DEC_8);
    iis3dwb_timestamp_set(&dev, PROPERTY_ENABLE);

    // Enable INT1 for threshold (watermark)
    iis3dwb_pin_int1_route_t int1 = {};
    int1.fifo_th = 1;
    iis3dwb_pin_int1_route_set(&dev, &int1);

    // ISR install
    gpio_config_t interrupt_pin = {
        .intr_type = GPIO_INTR_POSEDGE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1 << SENSOR_INT1),
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en = GPIO_PULLUP_DISABLE
    };
    gpio_install_isr_service(0);
    gpio_config(&interrupt_pin);
    gpio_isr_handler_add(SENSOR_INT1, isr_sample, NULL);
}

// watermark interrupt
void sensor_read(void)
{
    static uint8_t tx_buffer[1000];

    iis3dwb_fifo_status_t fifo_status;
    iis3dwb_fifo_status_get(&dev, &fifo_status);
    uint16_t num = fifo_status.fifo_level;

    iis3dwb_fifo_out_multi_raw_get(&dev_ctx, fifo_data, num);

    for (uint16_t k = 0; k < num; k++) {
        iis3dwb_fifo_out_raw_t *sample = &fifo_data[k];

        switch (sample->tag >> 3) {
            case IIS3DWB_XL_TAG:
                int16_t *x = &sample->data[0];
                int16_t *y = &sample->data[2];
                int16_t *z = &sample->data[4];
            
                sprintf(buffer, "%d\t%4.2f\t%4.2f\t%4.2f\r\n",
                        k,
                        iis3dwb_from_fs2g_to_mg(*x),
                        iis3dwb_from_fs2g_to_mg(*y),
                        iis3dwb_from_fs2g_to_mg(*z));
                // write buffer [mg] to file
                break;
            case IIS3DWB_TIMESTAMP_TAG:
                int32_t *ts = &sample->data[0];
                sprintf(buffer, "%d\t%d\r\n", k, *ts);
                // write timestamp [ms] to file
                break;
            default:
                break;
        }
    }
}


// ISR for switch ON/OFF  - GPIO + handler
// ISR for INT1


void app_main(void)
{
    sdmmc_card_t *card = storage_enable(MOUNT_POINT);
    // Find filename - list directory - next file in sequence
    if (stat(LOG_DIR, &st) == -1) {
        mkdir(LOG_DIR, 0755);
    }
    unsigned long file_seq = get_new_recording_name(LOG_DIR);


    // Create file
    const char path[MAX_FILENAME];
    snprintf(path, MAX_FILENAME, "%s/%d.csv", LOG_FOLDER, file_seq);
    FILE *log = fopen(path, "w");
    if (f == NULL) {
        return ESP_FAIL;
    }

    // Check if file still exists & Write to file 
    struct stat st;
    if (stat(path, &st) == 0) {
        fprintf(log, "Test\n");
    }

    // Close file
    fclose(log);
    storage_disable(card, MOUNT_POINT);
    

    sensor_enable();
    // Wait for switch on, swich off from interrupt
    xTaskCreate(sampler_task, "sampling", 1024, NULL, 1, &sample_tick);
    while(1);
}
