#include "pinout.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"


TaskHandle_t trigger_task;
TaskHandle_t sampler_task;
QueueHandle_t samples;

stmdev_ctx_t sensor;
sdmmc_card_t *card = NULL;

// Write only from read_accelerometer when sensor is enabled
int32_t sensor_timestamp = 0;

SemaphoreHandle_t file_mutex;
FILE *file = NULL;

void panic(int delay)
{
    bool status = true;
    while (true) {
        led_light(status);
        vTaskDelay(delay / portTICK_PERIOD_MS);
        status = !status;
    }
}


static void isr_sample(void* args)
{
    xTaskNotifyGive(sampler_task);
}

static void IRAM_ATTR isr_switch(void *args)
{
    BaseType_t higher_priority_woken = pdFALSE;
    vTaskNotifyGiveFromISR(trigger_task, &higher_priority_woken);
    portYIELD_FROM_ISR(higher_priority_woken);
}

void push_trigger(void *args)
{
    bool is_recording = false;
    char filename[MAX_FILENAME];
    spi_device_handle_t spi;

    const esp_timer_create_args_t sampler_timer_conf = {
        .callback = &isr_sample,
        .name = "sampler"
    };
    esp_timer_handle_t sampler_timer;
    esp_timer_create(&sampler_timer_conf, &sampler_timer);

    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            if (!is_recording) {
                // Start recording
                switch_disable();

                // Open file
                get_recording_filename(filename, LOG_FOLDER);
                if (xSemaphoreTake(file_mutex, portMAX_DELAY) == pdTRUE) {
                    file = fopen(filename, "w");
                    if (file == NULL) {
                        panic(200);
                    }
                    xSemaphoreGive(file_mutex);
                }
                // Run recorder
                sensor_timestamp = 0;
                sensor_enable(&spi, &sensor);
                vTaskDelay(10 / portTICK_PERIOD_MS);
                esp_timer_start_periodic(sampler_timer, SAMPLE_RATE);

                led_light(true);
                // Debounce delay for switch
                vTaskDelay(2000 / portTICK_PERIOD_MS);

                is_recording = true;
                switch_enable(false, isr_switch);

            } else {
                // Stop recording
                switch_disable();
                // Stop recorder
                esp_timer_stop(sampler_timer);
                // wait for transactions to end
                vTaskDelay(10 / portTICK_PERIOD_MS);
                sensor_disable(spi);

                // Close file
                if (xSemaphoreTake(file_mutex, portMAX_DELAY) == pdTRUE) {
                    if (file != NULL) {
                        fclose(file);
                    }
                    file = NULL;
                    xSemaphoreGive(file_mutex);
                }

                led_light(false);
                // Debounce delay for switch
                vTaskDelay(2000 / portTICK_PERIOD_MS);
                is_recording = false;
                switch_enable(true, isr_switch);
            }
        }
    }
}

void read_accelerometer(void *args)
{
    static iis3dwb_fifo_out_raw_t fifo_data[FIFO_LENGTH];
    AccResolutionConvert conv = &iis3dwb_from_fs2g_to_mg;

    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            iis3dwb_fifo_status_t fifo_status;
            iis3dwb_fifo_status_get(&sensor, &fifo_status);
            uint16_t num = fifo_status.fifo_level;
            // (num == FIFO_LENGTH)

            iis3dwb_fifo_out_multi_raw_get(&sensor, fifo_data, num);

            for (uint16_t k = 0; k < num; k++) {
                iis3dwb_fifo_out_raw_t *sample = &fifo_data[k];
                Acceleration acc;

                switch (sample->tag >> 3) {
                    case IIS3DWB_XL_TAG:
                        acc.x = conv(*(int16_t *)&sample->data[0]);
                        acc.y = conv(*(int16_t *)&sample->data[2]);
                        acc.z = conv(*(int16_t *)&sample->data[4]);
                        acc.t = sensor_timestamp;
                        xQueueSend(samples, &acc, WAIT_TICKS);
                        // (BaseType_t err == errQUEUE_FULL)
                        break;
                    case IIS3DWB_TIMESTAMP_TAG:
                        sensor_timestamp = *(int32_t *)&sample->data[0];
                        break;
                    default:
                        break;
                }
            }
        }
    }
}


void write_card(void *args)
{
    // Resolution: 2g, [mg] units
    Acceleration acc;

    while (true) {
        if (xQueueReceive(samples, &acc, portMAX_DELAY) == pdTRUE) {
            if (xSemaphoreTake(file_mutex, WAIT_TICKS) == pdTRUE) {
                if (file != NULL) {
                    fprintf(file, "%ld\t%4.2f\t%4.2f\t%4.2f\n", acc.t, acc.x, acc.y, acc.z);
                }
                xSemaphoreGive(file_mutex);
            }
        }
    }
}


void app_main(void)
{
    file_mutex = xSemaphoreCreateMutex();
    samples = xQueueCreate(QUEUE_LENGTH, sizeof(Acceleration));
    gpio_install_isr_service(0);

    card = storage_enable(MOUNT_POINT);
    if (card == NULL) {
        panic(500);
    }

    led_enable();
    switch_enable(true, isr_switch);

    xTaskCreate(push_trigger, "trigger", 4096, NULL, 2, &trigger_task);
    xTaskCreate(read_accelerometer, "read", 4096, NULL, 1, &sampler_task);
    xTaskCreate(write_card, "write", 8192, NULL, 1, NULL);
}
