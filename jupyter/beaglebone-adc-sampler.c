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
// ODR = 500 Hz
#define FS_HZ               4000
#define PERIOD_US           (US_PER_SECOND / (double)FS_HZ)

#define ANALOG_IN_PATH      "/sys/bus/iio/devices/iio:device0/"
#define AIN_CH              3

#define AIN0                ANALOG_IN_PATH "in_voltage0_raw"
#define AIN1                ANALOG_IN_PATH "in_voltage1_raw"
#define AIN2                ANALOG_IN_PATH "in_voltage2_raw"
#define AIN3                ANALOG_IN_PATH "in_voltage3_raw"
#define AIN4                ANALOG_IN_PATH "in_voltage4_raw"
#define AIN5                ANALOG_IN_PATH "in_voltage5_raw"
#define AIN6                ANALOG_IN_PATH "in_voltage6_raw"
#define BUF_SIZE            8192


// Time events
int make_periodic(unsigned int period)
{
    int fd = timerfd_create(CLOCK_MONOTONIC, 0);
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

    int ret = read(timer, &missed, sizeof(missed));
    if (ret == -1) {
        perror("read timer");
        return;
    }
}


// Measure time intervaô
uint64_t sub_timespec(struct timespec *a, struct timespec *b)
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

struct timespec write_interval_ns(int log_fd, struct timespec before)
{
    char s[20];
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);
    snprintf(s, 20, "%llu", sub_timespec(&before, &now) / 1000);
    write(log_fd, s, strlen(s));
    return now;
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
    struct timespec t = {};

    // period in nanoseconds
    const char *ain_channels[AIN_CH] = {AIN0, AIN2, AIN6};
    puts("Sampler");

    char buffer[BUF_SIZE];
    ssize_t ret;
    int log = open(argv[1], O_WRONLY | O_CREAT, 0644);

    puts("Recording to file ...");

    int timer = make_periodic(PERIOD_US);
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
        ///////////////////////////////////////////
        t = write_interval_ns(log, t);
        ////////////////////////////////////////////
        write(log, "\n", 1);
        samples++;
        wait_period(timer);
    }

    close(log);
    printf("Writen: %llu samples\n", samples);
    puts("Finish!");
}
