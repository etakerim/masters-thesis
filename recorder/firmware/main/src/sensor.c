#include "hal/gpio_types.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "iis3dwb.h"
#include "pinout.h"


static int32_t platform_write(void *handle, uint8_t reg, const uint8_t *buffer, uint16_t len)
{
    spi_transaction_t t = {
        .addr = reg & 0x7F,
        .length = 8 * len,
        .tx_buffer = buffer
    };
    spi_device_handle_t spi = *(spi_device_handle_t *)handle;
    spi_device_transmit(spi, &t);
    return 0;
}

static int32_t platform_read(void *handle, uint8_t reg, uint8_t *buffer, uint16_t len)
{
    uint8_t tx_buffer[1] = {reg | 0x80};

    spi_transaction_t t = {
        .length = 8 * sizeof(tx_buffer),
        .rxlength = 8 * len,
        .tx_buffer = tx_buffer,
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
        .max_transfer_sz = 1024
    };
    err = spi_bus_initialize(SPI_BUS, &spi_bus, SPI_DMA_DISABLED);
    spi_device_interface_config_t spi_iface = {
        .clock_speed_hz=SPI_BUS_FREQUENCY,
        .flags=SPI_DEVICE_HALFDUPLEX,
        .address_bits=7,
        .mode=0,
        .spics_io_num=SENSOR_CS,
        .queue_size=3
    };
    err = spi_bus_add_device(SPI_BUS, &spi_iface, spi_dev);

    return err;
}

void sensor_enable(stmdev_ctx_t *dev)
{
    spi_device_handle_t spi_dev;
    spi_enable(&spi_dev);

    dev->write_reg = platform_write;
    dev->read_reg = platform_read;
    dev->handle = &spi_dev;

    uint8_t who_am_i;
    iis3dwb_device_id_get(dev, &who_am_i);
    if (who_am_i != IIS3DWB_ID)
        while (1);

    iis3dwb_reset_set(dev, PROPERTY_ENABLE);

    uint8_t rst;
    do {
        iis3dwb_reset_get(dev, &rst);
    } while (rst);

    iis3dwb_block_data_update_set(dev, PROPERTY_ENABLE);

    iis3dwb_xl_full_scale_set(dev, IIS3DWB_2g);
    iis3dwb_fifo_watermark_set(dev, FIFO_WATERMARK);
    iis3dwb_fifo_xl_batch_set(dev, IIS3DWB_XL_BATCHED_AT_26k7Hz);
    iis3dwb_fifo_mode_set(dev, IIS3DWB_STREAM_MODE);

    iis3dwb_xl_data_rate_set(dev, IIS3DWB_XL_ODR_26k7Hz);
    iis3dwb_fifo_timestamp_batch_set(dev, IIS3DWB_DEC_8);
    iis3dwb_timestamp_set(dev, PROPERTY_ENABLE);
}

void sensor_int_threshold_enable(stmdev_ctx_t *dev, gpio_isr_t isr_handler)
{
    iis3dwb_pin_int1_route_t int1 = {};
    int1.fifo_th = 1;
    iis3dwb_pin_int1_route_set(dev, &int1);

    gpio_config_t interrupt_pin = {
        .intr_type = GPIO_INTR_POSEDGE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL << SENSOR_INT1),
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en = GPIO_PULLUP_DISABLE
    };
    gpio_config(&interrupt_pin);
    gpio_isr_handler_add(SENSOR_INT1, isr_handler, NULL);
}

void sensor_int_threshold_disable(void)
{
    gpio_isr_handler_remove(SENSOR_INT1);
}


void sensor_read(stmdev_ctx_t *dev, FILE *output)
{
    static iis3dwb_fifo_out_raw_t fifo_data[FIFO_WATERMARK];

    iis3dwb_fifo_status_t fifo_status;
    iis3dwb_fifo_status_get(dev, &fifo_status);
    uint16_t num = fifo_status.fifo_level;

    iis3dwb_fifo_out_multi_raw_get(dev, fifo_data, num);

    for (uint16_t k = 0; k < num; k++) {
        iis3dwb_fifo_out_raw_t *sample = &fifo_data[k];

        switch (sample->tag >> 3) {
            case IIS3DWB_XL_TAG:
                int16_t *x = (int16_t *)&sample->data[0];
                int16_t *y = (int16_t *)&sample->data[2];
                int16_t *z = (int16_t *)&sample->data[4];
            
                fprintf(output, "%d\t%4.2f\t%4.2f\t%4.2f\r\n",
                        k,
                        iis3dwb_from_fs2g_to_mg(*x),
                        iis3dwb_from_fs2g_to_mg(*y),
                        iis3dwb_from_fs2g_to_mg(*z));
                // [mg] units
                break;
            case IIS3DWB_TIMESTAMP_TAG:
                int32_t *ts = (int32_t *)&sample->data[0];
                fprintf(output, "%d\t%ld\r\n", k, *ts);
                // [ms] units
                break;
            default:
                break;
        }
    }
}
