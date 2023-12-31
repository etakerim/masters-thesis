Čerpadlá:
- KSB 1 nové (pos. 1, 400 kW, 2018, 1489 rpm)
- KSB 2 nové (pos. 2, 400 kW)
- Sigma väčšia (400l/s, 420 kW, 1485 rpm)

- 02/2024
- 03/2024
- 04/2024

- profylatktický servis (každý rok)
- viac ako 5 rokov ložiská sa nevymieňajú sa
- sigma rotácie 1485 rpm


```
    git clone -b v5.1.2 --recursive https://github.com/espressif/esp-idf.git esp-idf-v5.1.2
    cd esp-idf-v5.1.2/
    ./install.sh esp32
# /home/miroslav/.espressif/


    . ./export.sh

    cp -r $IDF_PATH/examples/get-started/hello_world .
    idf.py set-target esp32
    idf.py menuconfig
    idf.py build
    idf.py -p /dev/ttyUSB0 flash
```