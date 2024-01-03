#include "pinout.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"


static TaskHandle_t log_start;
static TaskHandle_t sample_tick;
static FILE *measurement = NULL;  // TODO: Protect by mutex
static stmdev_ctx_t sensor;

static const char *TAG = "main";


static void IRAM_ATTR isr_sample(void *args)
{
    BaseType_t higher_priority_woken = pdFALSE;
    vTaskNotifyGiveFromISR(sample_tick, &higher_priority_woken);
}

static void IRAM_ATTR isr_switch(void *args)
{
    BaseType_t higher_priority_woken = pdFALSE;
    vTaskNotifyGiveFromISR(log_start, &higher_priority_woken);
}

void push_to_record(void *args)
{
    bool is_recording = false;
    char filename[MAX_FILENAME];
    sdmmc_card_t *card = NULL;
    gpio_install_isr_service(0);

    led_enable();
    switch_enable(true, isr_switch);
    ESP_LOGI(TAG, "Switch enabled");

    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            if (!is_recording) {   // start recording
                ESP_LOGI(TAG, "Start recording");
                switch_disable();

                card = storage_enable(MOUNT_POINT);

                get_recording_filename(filename, LOG_FOLDER);
                measurement = fopen(filename, "w");
                if (measurement == NULL) {
                    ESP_LOGI(TAG, "FILE NOT OPEN!");
                }

                // Measure ODR (for 1 second) - Chapter 7.2 of docs

                /* TODO - testing 
                sensor_enable(&sensor);
                sensor_int_threshold_enable(&sensor, isr_sample);
                */

            
                led_light(true);
                vTaskDelay(2000 / portTICK_PERIOD_MS);    // Debounce delay for switch 
                is_recording = true;
                switch_enable(false, isr_switch);
                ESP_LOGI(TAG, "Start recording - can stop");

                // DEBUG
                for (float i = 0; i < 100; i++) {
                    fprintf(measurement, "%d\t%4.2f\t%4.2f\t%4.2f\r\n",
                        (int)i,
                        2*i,
                        2*i+1,
                        2*i+2);
                }

            } else {            // stop recording   
                ESP_LOGI(TAG, "Stop recording");
                switch_disable();
                /* TODO - testing
                sensor_int_threshold_disable();
                */
                
                fclose(measurement);             // TODO: log protect with mutex
                storage_disable(card, MOUNT_POINT);
                ESP_LOGI(TAG, "Unmount card");

                led_light(false);
                vTaskDelay(2000 / portTICK_PERIOD_MS);    // Debounce delay for switch 
                is_recording = false;
                switch_enable(true, isr_switch);

                ESP_LOGI(TAG, "Start recording - can start");
            }  
        }
    }
}

void sampler_task(void *args)
{
    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            sensor_read(&sensor, measurement);
        }
    }
}


void app_main(void)
{
    xTaskCreatePinnedToCore(push_to_record, "rec", 8192, NULL, 1, &log_start, tskNO_AFFINITY);
    // TODO: xTaskCreatePinnedToCore(sampler_task, "write", 4096, NULL, 1, &sample_tick, tskNO_AFFINITY);
}
