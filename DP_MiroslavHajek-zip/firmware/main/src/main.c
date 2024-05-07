#include "pinout.h"

// RTOS primitives
TaskHandle_t trigger_task;
TaskHandle_t sampler_task;
QueueHandle_t samples;

// HW device
spi_device_handle_t spi;
stmdev_ctx_t sensor;
sdmmc_card_t *card = NULL;

// File and its mutex
SemaphoreHandle_t file_mutex;
FILE *file = NULL;

// Is recording flag
bool is_recording = false;
// Write only from read_accelerometer when sensor is enabled
int32_t sensor_timestamp = 0;


// Sampler timer
static void isr_sample(void* args)
{
    xTaskNotifyGive(sampler_task);
}

const esp_timer_create_args_t sampler_timer_conf = {
        .callback = &isr_sample
};
esp_timer_handle_t sampler_timer;


static void IRAM_ATTR isr_switch(void *args)
{
    BaseType_t higher_priority_woken = pdFALSE;
    vTaskNotifyGiveFromISR(trigger_task, &higher_priority_woken);
    portYIELD_FROM_ISR(higher_priority_woken);
}

// Stop timer
static void stop_timer_run(void* args)
{
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
    vTaskDelay(SWITCH_DEBOUNCE);
    is_recording = false;
    switch_enable(true, isr_switch);
}


// Auto turn off timer
const esp_timer_create_args_t stop_timer_conf = {
    .callback = &stop_timer_run
};
esp_timer_handle_t stop_timer;


void push_trigger(void *args)
{
    char filename[MAX_FILENAME];

    esp_timer_create(&sampler_timer_conf, &sampler_timer);
    esp_timer_create(&stop_timer_conf, &stop_timer);

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

                esp_timer_start_once(stop_timer, AUTO_TURN_OFF_US);
                esp_timer_start_periodic(sampler_timer, SAMPLE_RATE);

                led_light(true);
                vTaskDelay(SWITCH_DEBOUNCE);
                is_recording = true;
                switch_enable(false, isr_switch);

            } else {
                // Stop recording
                switch_disable();
                // Stop recorder
                esp_timer_stop(stop_timer);
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
                vTaskDelay(SWITCH_DEBOUNCE);
                is_recording = false;
                switch_enable(true, isr_switch);
            }
        }
    }
}

void read_accelerometer(void *args)
{
    Acceleration acc;
    iis3dwb_fifo_out_raw_t fifo_data[FIFO_LENGTH];

    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            iis3dwb_fifo_status_t fifo_status;
            iis3dwb_fifo_status_get(&sensor, &fifo_status);
            uint16_t n = fifo_status.fifo_level;

            if (n >= FIFO_LENGTH - 1) {
                led_light(false);
            }
            iis3dwb_fifo_out_multi_raw_get(&sensor, fifo_data, n);
            acc.len = 0;

            for (uint16_t k = 0; k < n; k++) {
                iis3dwb_fifo_out_raw_t *sample = &fifo_data[k];

                switch (sample->tag >> 3) {
                    case IIS3DWB_XL_TAG:
                        acc.t[acc.len] = sensor_timestamp;
                        acc.x[acc.len] = *(int16_t *)&sample->data[0];
                        acc.y[acc.len] = *(int16_t *)&sample->data[2];
                        acc.z[acc.len] = *(int16_t *)&sample->data[4];
                        acc.len++;
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
    Acceleration acc;
    int32_t buffer[NUM_OF_FIELDS * FIFO_LENGTH];

    while (true) {
        if (xQueueReceive(samples, &acc, portMAX_DELAY) == pdTRUE) {
            size_t idx = 0;
            for (uint16_t k = 0; k < acc.len; k++) {
                buffer[idx++] = acc.t[k];
                buffer[idx++] = acc.x[k];
                buffer[idx++] = acc.y[k];
                buffer[idx++] = acc.z[k];
            }

            if (xSemaphoreTake(file_mutex, NO_WAIT) == pdTRUE) {
                if (file != NULL) {
                    fwrite(&buffer, sizeof(int32_t), idx, file);
                    fflush(file);
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
    led_enable();

    card = storage_enable(MOUNT_POINT);
    if (card == NULL) {
        panic(500);
    }

    switch_enable(true, isr_switch);

    xTaskCreatePinnedToCore(push_trigger, "trigger", 4000, NULL, 2, &trigger_task, 1);
    xTaskCreatePinnedToCore(write_card, "write", 32000, NULL, 1, NULL, 1);
    xTaskCreatePinnedToCore(read_accelerometer, "read", 16000, NULL, 1, &sampler_task, 0);
}
