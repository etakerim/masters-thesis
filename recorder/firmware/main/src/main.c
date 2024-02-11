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
                //esp_timer_start_once(sampler_timer, SAMPLE_RATE);

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
    static Acceleration acc;

    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            iis3dwb_fifo_status_t fifo_status;
            iis3dwb_fifo_status_get(&sensor, &fifo_status);
            acc.len = fifo_status.fifo_level;

            if (acc.len >= FIFO_LENGTH - 1) {
                led_light(false);
            }
            iis3dwb_fifo_out_multi_raw_get(&sensor, fifo_data, acc.len);

            for (uint16_t k = 0; k < acc.len; k++) {
                iis3dwb_fifo_out_raw_t *sample = &fifo_data[k];

                switch (sample->tag >> 3) {
                    case IIS3DWB_XL_TAG:
                        acc.x[k] = *(int16_t *)&sample->data[0];
                        acc.y[k] = *(int16_t *)&sample->data[2];
                        acc.z[k] = *(int16_t *)&sample->data[4];
                        acc.t[k] = sensor_timestamp;
                        break;
                    case IIS3DWB_TIMESTAMP_TAG:
                        sensor_timestamp = *(int32_t *)&sample->data[0];
                        break;
                    default:
                        break;
                }
            }
            if (xQueueSend(samples, &acc, NO_WAIT) != pdPASS) {
                led_light(false);
            }
        }
    }
}




void write_card(void *args)
{
    static Acceleration acc;
    static int32_t buffer[4 * FIFO_LENGTH];

    //TickType_t initial_time = 0, end_time = 0;


    while (true) {
        if (xQueueReceive(samples, &acc, portMAX_DELAY) == pdTRUE) {
            //initial_time = xTaskGetTickCount();
            for (uint16_t k = 0; k < acc.len; k++) {
                buffer[4*k + 0] = acc.t[k];
                buffer[4*k + 1] = acc.x[k];
                buffer[4*k + 2] = acc.y[k];
                buffer[4*k + 3] = acc.z[k];
                // TODO: write to text buffer??
            }

            if (xSemaphoreTake(file_mutex, NO_WAIT) == pdTRUE) {
                if (file != NULL) {
                    fwrite(&buffer, sizeof(int16_t), acc.len, file);
                    fflush(file);
                }
                xSemaphoreGive(file_mutex);
            }
            //end_time = xTaskGetTickCount();
            //ESP_LOGW("m", "%lu", end_time - initial_time);
        }
    }
}


void app_main(void)
{
    file_mutex = xSemaphoreCreateMutex();
    samples = xQueueCreate(QUEUE_LENGTH, sizeof(Acceleration));
    gpio_install_isr_service(0);
    led_enable();

    card = storage_enable(MOUNT_POINT);
    if (card == NULL) {
        panic(500);
    }

    switch_enable(true, isr_switch);

    xTaskCreatePinnedToCore(push_trigger, "trigger", 4096, NULL, 4, &trigger_task, 1);
    xTaskCreatePinnedToCore(write_card, "write", 8192, NULL, 2, NULL, 1);
    xTaskCreatePinnedToCore(read_accelerometer, "read", 4096, NULL, 1, &sampler_task, 0);
}
