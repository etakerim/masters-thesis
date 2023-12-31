#include <stdio.h>
#include <string.h>
#include <sys/unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"
#include "pinout.h"


sdmmc_card_t *storage_enable(const char *mount_point)
{
    esp_vfs_fat_sdmmc_mount_config_t mount = {
        .format_if_mount_failed = true,     // Set to false
        .max_files = 5,                     // Maximum number of opened files
        .allocation_unit_size = 16 * 1024   // Useful only for format
    };

    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();  // TODO: SDMMC_HOST_SLOT_1
    slot.width = 1;

    sdmmc_card_t *card;
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    esp_err_t ret = esp_vfs_fat_sdmmc_mount(mount_point, &host, &slot, &mount, &card);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            // Failed to mount filesystem
        } else {
            // Failed to initialize the card
        }
        return NULL;
    }

    // Card has been initialized, print its properties
    // sdmmc_card_print_info(stdout, card);
    // Format FATFS
    // ret = esp_vfs_fat_sdcard_format(mount_point, card);

    return card;
}

void storage_disable(sdmmc_card_t *card, const char *mount_point) 
{
    esp_vfs_fat_sdcard_unmount(mount_point, card);
}

static unsigned long get_new_recording_name(const char *path) 
{
    uint32_t seq = 1;
    char filename[MAX_FILENAME];

    DIR *folder = opendir(path);
    if (folder == NULL)
        return seq;
    
    struct dirent *entry;

    while ((entry = readdir(folder)) != NULL) {
        strncpy(filename, entry->d_name, MAX_FILENAME);
        char *delim = strpbrk(filename, ".");
        if (delim == NULL)
            continue;
        *delim = '\0';

        char *end;
        uint32_t name = strtol(filename, &end, 10);
        if (name > seq)
            seq = name;
    }
        
    closedir(folder);
    return seq;
}

FILE *create_recording(const char *path)
{
    struct stat st;
    if (stat(path, &st) < 0) {
        mkdir(path, 0755);
    }

    unsigned long file_seq = get_new_recording_name(path);
    const char filename[MAX_FILENAME];

    snprintf(filename, MAX_FILENAME, "%s/%ld.csv", path, file_seq);
    return fopen(path, "w");
}