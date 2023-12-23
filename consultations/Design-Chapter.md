# Design

https://power-mi.com/content/vibration-analysis-electric-induction-motors

### kNN classification

#### Models-Batch/kNN.html


#### Machinery and measurement

Table and images

- KALORIK **Stand Fan** TKG VT 1037 C
    - height 125 cm, stable 60 cm cross base
    - 3 speed, 45 cm fan diameter, 3 propelers
    - 45 W (Class I)
-  AC unit VERTIV: **Scroll Compressor**:
    -  Copeland ZR16M3E-TWD-561
    -  Datacentre SHC3, Petržalka
    - 2900 RPM @ 50 Hz;
    - 9.7 kW (13 HP);
    - 380/420V; 25 A)
    -  2 units - Class I
    - Faulty parts: unbalance of the rotor, fault of the scroll, mechanical assembly loose, bearing loose
- **Water pump**: KSB Omega 300-560 B GB G F  - Single-stage axially split volute casing pump
    - (2018,
    - 1493 (1500) RPM @ 50 Hz;
    - 400 kW elektromotor;
    - power requirement: 380 kW,
    - Class II - ISO) - 1 unit
    - Pumping station Podunajské Biskupice
    - Faulty parts: misalignment, bearings
    - Placements:
        - Shaft A, outer bearing housing number 1, triaxial accelerometer, positioned 90° counterclokwise from zero, 
- **Water pump** Sigma Lutín - 1986
- Columns: t, x, y, z, label - worse, better, pump - good, warning, fault (based on ISO)

#### Sensors:

  - **UML diagram**:
      - Block diagram for HW connections:
          - Accel (SPI, INT1,2) -> ESP -> (SDIO) SD card
              - ESP-IDF
              - Detached Switch (INT [34] replace button) and LED (GPIO [5]) to indicate recording
              - **Storage API (SDMMC)** SD card: (GPIO15 [HS2_CMD], GPIO14 [HS2_CLK], GPIO2 [HS2_DATA0])
                  - https://docs.espressif.com/projects/esp-idf/en/v5.1.2/esp32/api-reference/storage/fatfs.html#using-fatfs-with-vfs-and-sd-cards
              - **Peripherals API (SPI Master Driver)** Accel: (3.3V, GND, MISO [16], MOSI [32], CLK [33], CS [13], INT1 [35], INT2 [36]) Sensor max. 10 MHz
                  - https://docs.espressif.com/projects/esp-idf/en/v5.1.2/esp32/api-reference/peripherals/spi_master.html
                  - Driver: https://github.com/STMicroelectronics/IIS3DWB-PID/
              - <!-- Most of ESP32's peripheral signals have a direct connection to their dedicated IO\_MUX pins. However, the signals can also be routed to any other available pins using the less direct GPIO matrix. If at least one signal is routed through the GPIO matrix, then all signals will be routed through it. SPI2: MISO (12), MOSI (13), CLK (14); SPI3: MISO (19), MOSI (23), CLK (18), CS (5) --> 

      - Activity: Firmware fuction:
          - start measurement (switch) -> config sd card -> config accel (FIFO continous mode, INT to read ODR) / -> int -> read FIFO accel -> write buffer to SD card
      - Activity: Pipeline for feature extraction and selection (online)
          1. Take chunk of samples
          2. Compute temporal domain features from whole chunk
          3. Compute spectral domain features from whole chunk with window size and Welch
          4. Compute magnitudes of feature vectors to achieve axis independence. -> Send from edge device with timestamp
          5. <- Send label to timestamp to edge device (after some time)
          6. Running Compute rank of product of running corr, f statistic, mutual infomation (normalization not neccessary, forgetting factor?)
          7. When rank is stable n iterations start sending only subset + features with immidiate close ranks (first 3 + ranks up to 40% of full rank)
	- Component: whole infrastructure
   
   #### EDA
- **Fan**
    - Mounting: front, back (x,y - radial, z - axial)
    - Time waveform
        - Amplitudes (max): **$\pm 5\;m /s^2$** (x, y = sides),  **$\pm 1\;m /s^2$** (z = forwards/backwards)
        - Statistical tests 
            - Normality test: Kolmogorov–Smirnov test: **Not normal distribution** ($p < 0.001$ on 1 second)
            - Normality visual test: Quantile-quantile plot on chosen recording
            - Stationarity test: Augmented Dickey–Fuller test: **Stationarity** ($p < 0.001$ on 1 second)
            - Stationarity visual test: Autocorrelation plot
    - Spectrogram ($f_s$ = 2.5 kHz, w = 4096)
        - Maximal frequency:
            - Speed: 1 = 18.46 Hz $\pm$ 0.61 Hz  ($\approx$ 1100 RPM) 0.5 - 1.5 m/s^2
            - Speed: 2 = 20.35 Hz $\pm$ 0.61 Hz ($\approx$ 1200 RPM)
            - Speed: 3 = 21.99 Hz $\pm$ 0.61 Hz ($\approx$ 1300 RPM)
- **Compressor**
    - Mounting: top (x,y - radial, z - axial), side (x,z - radial, y - axial)
    -  Amplitudes (max): **$\pm 3\;m /s^2$** (all axis)
    - AC unit: no.3 (worse - 4rd harmonic twice the amplitude 0.2 to 0.4, 1st harmonic 0.8 in both), no.5 (better)
    - Time waveform
            - Normality test: Kolmogorov–Smirnov test: **Not normal distribution** ($p < 0.001$ on 1 second)
            - Stationarity test: Augmented Dickey–Fuller test: **Stationarity** ($p < 0.001$ on 1 second)
    - Spectrogram
        - Maximal frequency: 47.74 Hz ($\approx$ 2800 RPM)
- **Pump bearing**
    - Mounting: inner (x,z- radial, y - axial), outer (x,z- radial, y - axial) 2 pumps
    -  Amplitudes (max): **$\pm 5\;m /s^2$** (x, y = sides),  **$\pm 1\;m /s^2$** (z = forwards/backwards)
    -  1st Harmonic: 0.10 (rich spectrum to 200 Hz and than around 600 Hz)
    - Time waveform
            - Normality test: Kolmogorov–Smirnov test: **Not normal distribution** ($p < 0.001$ on 1 second)
            - Stationarity test: Augmented Dickey–Fuller test: **Stationarity** ($p < 0.001$ on 1 second)
    - Spectrogram
        - Maximal frequency: 24.37 Hz ($\approx$ 1460 RPM)

 ### **Design semi-supervised feature selection**
- Feature selection -> KNN -> Infer labels -> Feature selection
