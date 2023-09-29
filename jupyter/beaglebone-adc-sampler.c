/*
 * ssh 192.168.7.2 -l debian
 * gcc sampler.c -o sampler
 * sudo nice -n -20 ./sampler out.tsv
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

/*
 * https://www.analog.com/en/products/adxl335.html
low-pass filtering for antialiasing and noise reduction.
Bandwidth (Hz) Capacitor (μF)
    1 4.7
    10 0.47
    50 0.10
    100 0.05
    200 0.027
    500 0.01
*/
// One second
// #define PERIOD_NS   1000000

// ODR = 500 Hz
// Sampling frequency = 2000 Hz
#define PERIOD_NS           500
#define ANALOG_IN_PATH      "/sys/bus/iio/devices/iio:device0/"
#define AIN_CH              3
#define AIN0                ANALOG_IN_PATH "in_voltage0_raw"
#define AIN1                ANALOG_IN_PATH "in_voltage1_raw"
#define AIN2                ANALOG_IN_PATH "in_voltage2_raw"
#define BUF_SIZE            8192


int make_periodic(unsigned int period)
{
    int fd = timerfd_create(CLOCK_REALTIME, 0);
    if (fd == -1)
        return fd;

    unsigned int sec = period / 1000000;
    unsigned int ns = (period - (sec * 1000000)) * 1000;

    struct itimerspec itval;
    itval.it_interval.tv_sec = sec;
    itval.it_interval.tv_nsec = ns;
    itval.it_value.tv_sec = sec;
    itval.it_value.tv_nsec = ns;

    timerfd_settime(fd, 0, &itval, NULL);
    return fd;
}

void wait_period(int timer)
{
    unsigned long long missed;

    int ret =  read(timer, &missed, sizeof(missed));
    if (ret == -1) {
        perror("read timer");
        return;
    }
}


volatile sig_atomic_t done = 0;

void term(int signum)
{
   done = 1;
}

int main(int argc, char* argv[])
{
    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = term;
    sigaction(SIGINT, &action, NULL);
    unsigned long long samples = 0;

    // period in nanoseconds
    const char *ain_channels[AIN_CH] = {AIN0, AIN1, AIN2};
    puts("Sampler");

    char buffer[BUF_SIZE];
    ssize_t ret;
    int log = open(argv[1], O_WRONLY | O_CREAT, 0644);
    puts("Recording to input file ...");

    int timer = make_periodic(PERIOD_NS);
    while (!done) {
        for (int i = 0; i < AIN_CH; i++) {
            int ain = open(ain_channels[i], O_RDONLY);
            if (ain < 0) {
                perror("ain");
                return 1;
            }
            ret = read(ain, buffer, BUF_SIZE);
            if (ret > 0) {
                write(log, &buffer, ret-1);
                write(log, "\t", 1);
            } else {
                perror("read");
            }
            close(ain);
        }
        write(log, "\n", 1);
        samples++;
        wait_period(timer);
    }

    close(log);
    printf("Sampling frequency: 2 kHz\n");
    printf("Writen: %llu samples\n", samples);
    puts("Finish!");
}

