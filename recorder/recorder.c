/*
 * debian:temppwd
 *
 * ssh debian@192.168.7.2
 * gcc sampler.c -o sampler
 *
 * sudo nice -n -20 ./sampler out.tsv
 * scp debian@192.168.7.2:/home/debian/out.tsv out.tsv
 *
 * FILE COLUMNS
 * x[raw]  y[raw]  z[raw]  time_interval[us]
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/timerfd.h>
#include <inttypes.h>


#define US_PER_SECOND       1000000
#define NS_PER_SECOND       1000000000

#define FS_HZ               8000
#define PERIOD_US           (US_PER_SECOND / (double)FS_HZ)

#define ANALOG_IN_PATH      "/sys/bus/iio/devices/iio:device0"
#define ANALOG_IN_DEV       "/dev/iio:device0"
#define AIN_CH              3

#define BUFFER_LENGTH_PATH  ANALOG_IN_PATH "/buffer/length"
#define BUFFER_ENABLE_PATH  ANALOG_IN_PATH "/buffer/enable"
#define AIN0                ANALOG_IN_PATH "/scan_elements/in_voltage0_en"
#define AIN1                ANALOG_IN_PATH "/scan_elements/in_voltage1_en"
#define AIN2                ANALOG_IN_PATH "/scan_elements/in_voltage2_en"
#define AIN3                ANALOG_IN_PATH "/scan_elements/in_voltage3_en"
#define AIN4                ANALOG_IN_PATH "/scan_elements/in_voltage4_en"
#define AIN5                ANALOG_IN_PATH "/scan_elements/in_voltage5_en"
#define AIN6                ANALOG_IN_PATH "/scan_elements/in_voltage6_en"
#define CNT_SAMPLES         100
#define BUF_SIZE            AIN_CH * CNT_SAMPLES


volatile sig_atomic_t done = 0;

void term(int signum)
{
   done = 1;
}

void adc_enable(void)
{
    FILE *ain;
    const char *channels[AIN_CH] = {AIN0, AIN2, AIN6};

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
    fprintf("%d", 2 * AIN_CH * CNT_SAMPLES)
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
    ain = fopen(BUFFER_ENABLE_PATH, "w");
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

void input_loop(const char *filename)
{
    FILE *stream = fopen(ANALOG_IN_DEV, "rb");
    FILE *log = fopen(filename, "wb");

    if (stream == NULL || log == NULL) {
        perror("Cannot open stream or file");
        exit(1);
    }

    uint64_t samples = 0;
    uint16_t buffer[BUF_SIZE];
    while (!done) {
        fread(&buffer, sizeof(uint16_t), BUF_SIZE, stream);
        fwrite(&buffer, sizeof(uint16_t), BUF_SIZE, log);
    }

    fclose(log);
    fclose(stream);
}

int main(int argc, char* argv[])
{
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = term;
    sigaction(SIGINT, &action, NULL);

    if (args < 2) {
        puts("Missing filename");
        return 1;
    }

    puts("Sampler");
    puts("Recording to file ...");
    puts("Press ^C to stop");

    adc_enable();
    input_loop(argv[1]);
    adc_disable();



    // TODO: Postprocess to csv
    /*
    for (int i = 0; i < AIN_CH * CNT_SAMPLES; i += AIN_CH) {
        uint16_t x, y, z;
        fread(&x, sizeof(uint16_t), 1, stream);
        fread(&y, sizeof(uint16_t), 1, stream);
        fread(&z, sizeof(uint16_t), 1, stream);
        fprintf("%d\t%d\t%d\n", x, y, z);
                samples++;
    }
    */
    // https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html

    printf("Writen: %llu samples\n", samples);
    puts("Finish!");
}

