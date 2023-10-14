/*
 * debian:temppwd
 *
 * scp recorder.c debian@192.168.7.2:/home/debian/recorder.c
 * ssh debian@192.168.7.2
 *
 * FS = 2560 Hz (<2580)
 * gcc -Wall -Wextra -O2 recorder.c -o recorder
 * timeout --signal=SIGINT 10s ./recorder out
 *
 * scp debian@192.168.7.2:/home/debian/out.tsv out.tsv
 *
 * FILE COLUMNS
 * x[raw]  y[raw]  z[raw]
 *
 * Manuals:
 * https://elinux.org/EBC_Exercise_10a_Analog_In
 * https://www.kernel.org/doc/Documentation/ABI/testing/sysfs-bus-iio
 * https://www.kernel.org/doc/Documentation/devicetree/bindings/input/touchscreen/ti-tsc-adc.txt
 *
 * TI-am335x-adc.0.auto
 * https://software-dl.ti.com/processor-sdk-linux/esd/docs/latest/linux/Foundational_Components/Kernel/Kernel_Drivers/ADC.html
 *
 * ADC config:
 * /opt/source/bb.org-overlays/src/arm/BB-ADC-00A0.dts
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/timerfd.h>
#include <inttypes.h>


#define NS_PER_SECOND       1000000000.0
#define ANALOG_IN_PATH      "/sys/bus/iio/devices/iio:device0"
#define ANALOG_IN_DEV       "/dev/iio:device0"

#define BUFFER_LENGTH_PATH  ANALOG_IN_PATH "/buffer/length"
#define BUFFER_ENABLE_PATH  ANALOG_IN_PATH "/buffer/enable"
#define AIN0                ANALOG_IN_PATH "/scan_elements/in_voltage0_en"
#define AIN1                ANALOG_IN_PATH "/scan_elements/in_voltage1_en"
#define AIN2                ANALOG_IN_PATH "/scan_elements/in_voltage2_en"
#define AIN3                ANALOG_IN_PATH "/scan_elements/in_voltage3_en"
#define AIN4                ANALOG_IN_PATH "/scan_elements/in_voltage4_en"
#define AIN5                ANALOG_IN_PATH "/scan_elements/in_voltage5_en"
#define AIN6                ANALOG_IN_PATH "/scan_elements/in_voltage6_en"

#define AIN_CH              3
#define CNT_SAMPLES         256
#define BUF_SIZE            AIN_CH * CNT_SAMPLES
#define BUFFER_ALLOC        3 * BUF_SIZE

#define FILENAME_LENGTH     200


static const char *channels[AIN_CH] = {AIN0, AIN2, AIN6};
volatile sig_atomic_t done = 0;

void term(int signum)
{
   done = 1;
}

void adc_enable(void)
{
    FILE *ain;

    // Enable AIN
    for (int i = 0; i < AIN_CH; i++) {
        ain = fopen(channels[i], "w");
        if (ain != NULL) {
            putc('1', ain);
            fclose(ain);
        } else {
            perror("Already enabled ...");
        }
    }

    // Set buffer length
    ain = fopen(BUFFER_LENGTH_PATH, "w");
    if (ain == NULL) {
        perror("Setting buffer length ...");
        exit(1);
    }
    fprintf(ain, "%d", BUFFER_ALLOC);
    fclose(ain);

    // Enable continous mode
    ain = fopen(BUFFER_ENABLE_PATH, "w");
    if (ain == NULL) {
        perror("Enabling continous mode ...");
        exit(1);
    }
    putc('1', ain);
    fclose(ain);
}

void adc_disable()
{
    puts("Turning off input");

    //  Disable continous
    FILE *ain = fopen(BUFFER_ENABLE_PATH, "w");
     if (ain == NULL) {
        perror("Enabling continous mode ...");
        exit(1);
    }
    putc('0', ain);
    fclose(ain);

    // Disable AIN
    for (int i = 0; i < AIN_CH; i++) {
        ain = fopen(channels[i], "w");
        if (ain != NULL) {
            putc('0', ain);
            fclose(ain);
        } else {
            perror("Already enabled ...");
        }
    }
}

void set_extension(char *filename, const char *src, int buffer_len, const char *ext)
{
    memset(filename, '\0', buffer_len);
    strncpy(filename, src, buffer_len - strlen(ext));
    strcat(filename, ext);
}


uint64_t diff_timespec(struct timespec *a, struct timespec *b)
{
    struct timespec r;
    r.tv_nsec = b->tv_nsec - a->tv_nsec;
    r.tv_sec  = b->tv_sec - a->tv_sec;

    if (r.tv_sec > 0 && r.tv_nsec < 0) {
        r.tv_nsec += NS_PER_SECOND;
        r.tv_sec--;
    }
    else if (r.tv_sec < 0 && r.tv_nsec > 0) {
        r.tv_nsec -= NS_PER_SECOND;
        r.tv_sec++;
    }

    return NS_PER_SECOND * r.tv_sec + r.tv_nsec;
}



uint64_t input_loop(const char *filename)
{
    char dst[FILENAME_LENGTH];
    set_extension(dst, filename, FILENAME_LENGTH, ".bin");

    FILE *stream = fopen(ANALOG_IN_DEV, "rb");
    FILE *log = fopen(dst, "wb");

    if (stream == NULL || log == NULL) {
        perror("Cannot open stream or file");
        exit(1);
    }

    uint16_t buffer[BUF_SIZE];
    struct timespec start, stop;

    clock_gettime(CLOCK_MONOTONIC, &start);
    while (!done) {
        fread(&buffer, sizeof(uint16_t), BUF_SIZE, stream);
        fwrite(&buffer, sizeof(uint16_t), BUF_SIZE, log);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);

    fclose(log);
    fclose(stream);
    return diff_timespec(&start, &stop);
}


uint64_t save_to_csv(const char *filename)
{
    char src[FILENAME_LENGTH];
    char dst[FILENAME_LENGTH];

    set_extension(src, filename, FILENAME_LENGTH, ".bin");
    set_extension(dst, filename, FILENAME_LENGTH, ".tsv");

    FILE *samples = fopen(src, "rb");
    FILE *csv = fopen(dst, "w");
    uint64_t row;

    for (row = 0; !feof(samples); row++) {
        uint16_t v;
        fread(&v, sizeof(uint16_t), 1, samples);
        fprintf(csv, "%u", v);

        if ((row + 1) % AIN_CH != 0) {
            putc('\t', csv);
        } else {
            putc('\n', csv);
        }
    }

    fclose(samples);
    fclose(csv);

    return row / AIN_CH;
}


int main(int argc, char* argv[])
{
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = term;
    sigaction(SIGINT, &action, NULL);

    if (argc < 2) {
        puts("Missing filename");
        return 1;
    }

    puts("Sampler");
    puts("Recording to file ...");
    puts("Press ^C to stop");

    adc_enable();
    uint64_t elapsed_ns = input_loop(argv[1]);
    double elapsed_s = elapsed_ns / NS_PER_SECOND;
    adc_disable();

    uint64_t samples = save_to_csv(argv[1]);

    printf("Written: %llu samples\n", samples);
    printf("Duration: %.3f s\n", elapsed_s);
    printf("Sampling rate: %.3f Hz\n", samples / elapsed_s);
    puts("Finish!");
}
