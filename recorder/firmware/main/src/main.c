#include "pinout.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"


TaskHandle_t log_start;
TaskHandle_t sample_tick;

SemaphoreHandle_t file_mutex;
FILE *file = NULL;

spi_device_handle_t spi;
stmdev_ctx_t sensor;


static void IRAM_ATTR isr_sample(void *args)
{
    BaseType_t higher_priority_woken = pdFALSE;
    vTaskNotifyGiveFromISR(sample_tick, &higher_priority_woken);
    portYIELD_FROM_ISR(higher_priority_woken);
}

static void IRAM_ATTR isr_switch(void *args)
{
    BaseType_t higher_priority_woken = pdFALSE;
    vTaskNotifyGiveFromISR(log_start, &higher_priority_woken);
    portYIELD_FROM_ISR(higher_priority_woken);
}

void push_to_record(void *args)
{
    bool is_recording = false;
    char filename[MAX_FILENAME];
    sdmmc_card_t *card = NULL;
    gpio_install_isr_service(0);

    led_enable();
    switch_enable(true, isr_switch);
    int n = 0;

    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {

            if (!is_recording) {
                // Start recording
                switch_disable();

                // Open file
                card = storage_enable(MOUNT_POINT);
                get_recording_filename(filename, LOG_FOLDER);
                if (xSemaphoreTake(file_mutex, portMAX_DELAY) == pdTRUE) {
                    file = fopen(filename, "w");
                    if (file == NULL)
                        ESP_LOGI("main", "File does not exists");
                    xSemaphoreGive(file_mutex);
                }

                sensor_enable(&spi, &sensor);
                sensor_events_enable(&sensor, isr_sample);

                led_light(true);
                // Debounce delay for switch
                vTaskDelay(2000 / portTICK_PERIOD_MS);

                is_recording = true;
                switch_enable(false, isr_switch);


            } else {
                // Stop recording
                switch_disable();
                sensor_events_disable();
                sensor_disable(spi);
            
                // Close file
                if (xSemaphoreTake(file_mutex, portMAX_DELAY) == pdTRUE) {
                    if (file != NULL) {
                        fclose(file);
                    }
                    file = NULL;
                    xSemaphoreGive(file_mutex);
                }
                storage_disable(card, MOUNT_POINT);
                
                led_light(false);
                // Debounce delay for switch
                vTaskDelay(2000 / portTICK_PERIOD_MS);
                is_recording = false;
                switch_enable(true, isr_switch);

            }
        }
    }
}

void sampler_task(void *args)
{
    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            if (xSemaphoreTake(file_mutex, portMAX_DELAY) == pdTRUE) {
                if (file != NULL) {
                    sensor_read(&sensor, file);
                }
                xSemaphoreGive(file_mutex);
            }
        }
    }
}


void app_main(void)
{
    // https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/sdspi_share.html
    file_mutex = xSemaphoreCreateMutex();
    xTaskCreate(push_to_record, "trigger", 4096, NULL, 1, &log_start);  
    xTaskCreate(sampler_task, "read", 8192, NULL, 1, &sample_tick);
}
