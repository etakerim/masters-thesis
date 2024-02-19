#include "pinout.h"


void switch_enable(bool on, gpio_isr_t isr_handler)
{
    gpio_config_t pin = {
        .pin_bit_mask = (1ULL << RECORD_SWITCH_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .intr_type = on ? GPIO_INTR_NEGEDGE: GPIO_INTR_POSEDGE
    };
    gpio_config(&pin);
    gpio_isr_handler_add(RECORD_SWITCH_PIN, isr_handler, NULL);
}

void switch_disable(void)
{
    gpio_isr_handler_remove(RECORD_SWITCH_PIN);
}

void panic(int delay)
{
    bool status = true;
    while (true) {
        led_light(status);
        vTaskDelay(delay / portTICK_PERIOD_MS);
        status = !status;
    }
}

void led_enable(void)
{
    gpio_config_t pin = {
        .mode = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL << RECORD_LED_PIN)
    };
    gpio_config(&pin);
}

void led_light(bool on)
{
    gpio_set_level(RECORD_LED_PIN, on);
}