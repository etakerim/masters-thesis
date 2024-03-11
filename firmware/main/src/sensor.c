#include "hal/gpio_types.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "iis3dwb.h"
#include "pinout.h"


static int32_t platform_write(void *handle, uint8_t reg, const uint8_t *buffer, uint16_t len)
{
    spi_transaction_t t = {
        .addr = reg & 0x7f,
        .length = 8 * len,
        .tx_buffer = buffer
    };
    spi_device_handle_t spi = *(spi_device_handle_t *)handle;
    spi_device_transmit(spi, &t);
    return 0;
}

static int32_t platform_read(void *handle, uint8_t reg, uint8_t *buffer, uint16_t len)
{
    spi_transaction_t t = {
        .addr = reg | 0x80,
        .rxlength = 8 * len,
        .rx_buffer = buffer
    };
    spi_device_handle_t spi = *(spi_device_handle_t *)handle;
    spi_device_transmit(spi, &t);
    return 0;
}

static esp_err_t spi_enable(spi_device_handle_t *spi_dev)
{
    esp_err_t err;

    spi_bus_config_t spi_bus = {
        .miso_io_num = SENSOR_MISO,
        .mosi_io_num = SENSOR_MOSI,
        .sclk_io_num = SENSOR_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = SENSOR_SPI_LENGTH
    };
    err = spi_bus_initialize(SPI_BUS, &spi_bus, SPI_DMA_CH_AUTO); // SPI_DMA_DISABLED
    ESP_ERROR_CHECK(err);
    spi_device_interface_config_t spi_iface = {
        .clock_speed_hz=SPI_BUS_FREQUENCY,
        .flags=SPI_DEVICE_HALFDUPLEX,
        .address_bits=8,
        .mode=0,
        .spics_io_num=SENSOR_CS,
        .queue_size=3
    };
    err = spi_bus_add_device(SPI_BUS, &spi_iface, spi_dev);
    ESP_ERROR_CHECK(err);

    return err;
}

int sensor_enable(spi_device_handle_t *spi_dev, stmdev_ctx_t *dev)
{
    // INT1 handler on MCU
    spi_enable(spi_dev);

    dev->write_reg = platform_write;
    dev->read_reg = platform_read;
    dev->handle = spi_dev;

    uint8_t who_am_i;
    iis3dwb_device_id_get(dev, &who_am_i);
    if (who_am_i != IIS3DWB_ID) {
        return -1;
    }

    iis3dwb_reset_set(dev, PROPERTY_ENABLE);

    uint8_t rst;
    do {
        iis3dwb_reset_get(dev, &rst);
    } while (rst);

    iis3dwb_block_data_update_set(dev, PROPERTY_ENABLE);

    // Resolution: 4g
    iis3dwb_xl_full_scale_set(dev, IIS3DWB_4g);
    // iis3dwb_xl_full_scale_set(dev,  IIS3DWB_16g);

    iis3dwb_fifo_watermark_set(dev, FIFO_WATERMARK);
    iis3dwb_fifo_xl_batch_set(dev, IIS3DWB_XL_BATCHED_AT_26k7Hz);
    iis3dwb_fifo_mode_set(dev, IIS3DWB_STREAM_MODE);

    iis3dwb_xl_data_rate_set(dev, IIS3DWB_XL_ODR_26k7Hz);
    iis3dwb_fifo_timestamp_batch_set(dev, IIS3DWB_DEC_8);
    iis3dwb_timestamp_set(dev, PROPERTY_ENABLE);

    return 0;
}

void sensor_disable(spi_device_handle_t spi_dev)
{
    spi_bus_remove_device(spi_dev);
    spi_bus_free(SPI_BUS);
}

void sensor_events_enable(stmdev_ctx_t *dev)
{
    iis3dwb_pin_int1_route_t int1 = {.fifo_th = 1};
    iis3dwb_pin_int1_route_set(dev, &int1);
}

void sensor_events_disable(stmdev_ctx_t *dev)
{
    iis3dwb_pin_int1_route_t int1 = {};
    iis3dwb_pin_int1_route_set(dev, &int1);
}

