#include "pinout.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"


static TaskHandle_t log_start;
static TaskHandle_t sample_tick;
static FILE *measurement = NULL;  // TODO: Protect by mutex
static stmdev_ctx_t sensor;


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
    sdmmc_card_t *card = NULL;
    gpio_install_isr_service(0);

    led_enable();
    switch_enable(true, isr_switch);

    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            if (!is_recording) {   // start recording
                switch_disable();

                card = storage_enable(MOUNT_POINT);
                measurement = create_recording(LOG_FOLDER);
                // if (log == NULL) {}
                sensor_enable(&sensor);
                sensor_int_threshold_enable(&sensor, isr_sample);
            
                led_light(true);
                vTaskDelay(2000 / portTICK_PERIOD_MS);    // Debounce delay for switch 
                is_recording = true;
                switch_enable(false, isr_switch);

            } else {            // stop recording   
                switch_disable();
                sensor_int_threshold_disable();
                fclose(measurement);   // TODO: log protect with mutex
                storage_disable(card, MOUNT_POINT);

                led_light(false);
                vTaskDelay(2000 / portTICK_PERIOD_MS);    // Debounce delay for switch 
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
            sensor_read(&sensor, measurement);
        }
    }
}


void app_main(void)
{
    xTaskCreatePinnedToCore(push_to_record, "rec", 4096, NULL, 1, &log_start, tskNO_AFFINITY);
    xTaskCreatePinnedToCore(sampler_task, "write", 4096, NULL, 1, &sample_tick, tskNO_AFFINITY);
}
