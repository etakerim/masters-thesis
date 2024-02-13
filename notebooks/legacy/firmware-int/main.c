#include "pinout.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"


TaskHandle_t trigger_task;
TaskHandle_t sampler_task;
QueueHandle_t samples;

// DO mutex
stmdev_ctx_t sensor;
spi_device_handle_t spi;
sdmmc_card_t *card = NULL;

SemaphoreHandle_t file_mutex;
FILE *file = NULL;

static void IRAM_ATTR isr_sample(void *args)
{
    BaseType_t higher_priority_woken = pdFALSE;
    vTaskNotifyGiveFromISR(sampler_task, &higher_priority_woken);
    portYIELD_FROM_ISR(higher_priority_woken);
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

    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            if (!is_recording) {
                // Start recording
                switch_disable();

                // Open file
                get_recording_filename(filename, LOG_FOLDER);
                if (xSemaphoreTake(file_mutex, portMAX_DELAY) == pdTRUE) {
                    file = fopen(filename, "w");
                    if (file == NULL)
                        ESP_LOGI("main", "File does not exists");
                    xSemaphoreGive(file_mutex);
                }
                sensor_events_enable(&sensor);
            
                led_light(true);
                // Debounce delay for switch
                vTaskDelay(2000 / portTICK_PERIOD_MS);

                is_recording = true;
                switch_enable(false, isr_switch);

            } else {
                // Stop recording
                switch_disable();
                sensor_events_disable(&sensor);
            
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
    static iis3dwb_fifo_out_raw_t fifo_data[FIFO_WATERMARK];

    while (true) {
        if (ulTaskNotifyTake(pdTRUE, portMAX_DELAY) == pdTRUE) {
            iis3dwb_fifo_status_t fifo_status;
            iis3dwb_fifo_status_get(&sensor, &fifo_status);
            uint16_t num = fifo_status.fifo_level;
            //ESP_LOGI("m", "R");
            ESP_LOGI("m", "N=%d", num);
            iis3dwb_fifo_out_multi_raw_get(&sensor, fifo_data, num);
            
            for (uint16_t k = 0; k < num; k++) {
                iis3dwb_fifo_out_raw_t *sample = &fifo_data[k];
                Acceleration acc;

                switch (sample->tag >> 3) {
                    case IIS3DWB_XL_TAG:
                        acc.x = iis3dwb_from_fs2g_to_mg(*(int16_t *)&sample->data[0]);
                        acc.y = iis3dwb_from_fs2g_to_mg(*(int16_t *)&sample->data[2]);
                        acc.z = iis3dwb_from_fs2g_to_mg(*(int16_t *)&sample->data[4]);
                        xQueueSend(samples, &acc, WAIT_TICKS);
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
                    fprintf(file, "%4.2f\t%4.2f\t%4.2f\n",
                            acc.x, acc.y, acc.z);            // Resolution: 2g, [mg] units 
                }    
                xSemaphoreGive(file_mutex);
            }     
        }
    }
}


void app_main(void)
{
    file_mutex = xSemaphoreCreateMutex();
    samples = xQueueCreate(FIFO_WATERMARK * 3, sizeof(Acceleration));

    gpio_install_isr_service(0);
    gpio_config_t interrupt_pin = {
        .intr_type = GPIO_INTR_POSEDGE,
        .mode = GPIO_MODE_INPUT,
        .pin_bit_mask = (1ULL << SENSOR_INT1),
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en = GPIO_PULLUP_DISABLE
    };
    gpio_config(&interrupt_pin);
    gpio_isr_handler_add(SENSOR_INT1, isr_sample, NULL);

    sensor_enable(&spi, &sensor);
    card = storage_enable(MOUNT_POINT);
    if (card == NULL) {
        bool status = true;
        while (true) {
            led_light(status);
            vTaskDelay(500 / portTICK_PERIOD_MS);
            status = !status;
        }
    }

    led_enable();
    switch_enable(true, isr_switch);

    xTaskCreate(push_trigger, "trigger", 4096, NULL, 2, &trigger_task);  
    xTaskCreate(read_accelerometer, "read", 4096, NULL, 1, &sampler_task);    
    xTaskCreate(write_card, "write", 8192, NULL, 1, NULL);
}
