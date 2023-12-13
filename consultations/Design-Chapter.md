# Design in Masters Thesis

### Research questions
- Questions:
    - **Q1**: Which time and frequency features can be extracted from vibrational signals to provide an accurate record of machinery faults?
    - **Q2**: What are the savings in transmission bandwidth when chosen signal features are used in comparison to raw sampled measurement or lossless compression techniques?
    - **Q3**: How can the machinery faults be continuously identified based on collected events?
- Goals:
  1. How does vibration data looks like in temporal and spectral domain and what about extracted features
  2. Create dataset of machinery vibrations
  3. Reduce number of features (samples) send from edge device - to minimal amount
  4. Evaluate model performance on all features and different feature sets


### Dataset preprocessing

Files:
- EDA-Measure.ipynb

#### EDA
- **Mafaulda**
    - Time waveform
        - Amplitudes: **$\pm 3\;m /s^2$** (normal, misalign),  **$\pm 11\;m /s^2$** (severe faults)
        - Statistical tests
            - Normality test: Kolmogorov–Smirnov test: **Not normal distribution** ($p < 0.001$ on 1 second)
            - Stationarity visual test: Augmented Dickey–Fuller test: **Stationarity** ($p < 0.001$ on 1 second)
        - Spectrum
            - **Graph** from `Time domain waveform zoom - faults side by side`
            - **Graph** from `compare\_limited\_specrograms` axis y
            - Cumulative energy spectral plot - 80% of all energy in most faults under 4 - 6 kHz (more than 6 kHz for misalignement and normal)
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

#### MaFaulDa feature extraction:
- split each recording to 5 parts (1 second - 50000 samples)
- Remove mean - DC component filter
- Filter frequencies: cutoff = 10 kHz (Butterworth lowpass filter of 5 order) - peak 20 kHz (not possibe to capture)
- fault, severity, seq, rpm (out of impulse tachometer = prominence=3, width=50, 60 / $\Delta t$), features: ax/ay/az_feature_name
- spectral 5 window sizes: windows and welch averaging (2**14 = 16384 samples, 3 okná, 3.05 Hz resolution)
- TBD: Additional features:
    -**Find harmonics** (peaks with MMS and filter with threshold mean+2*std) and harmonic series
    -**WPD energy**, kurtosis

- Fault classes:
    - 'normal': 'normal',
    - 'imbalance': 'imbalance',
    - 'horizontal-misalignment': 'misalignment',
    - 'vertical-misalignment': 'misalignment',
    - 'overhang-cage\_fault': 'cage fault',
    - 'underhang-cage\_fault': 'cage fault',
    - 'underhang-ball\_fault': 'ball fault',
    - 'overhang-ball\_fault': 'ball fault',
    - 'overhang-outer\_race': 'outer race fault',
    - 'underhang-outer\_race': 'outer race fault'
- **Counts of classes** (to one table with 4 columns) - 1951 files (x 5 for split records) - **Unbalanced dataset**
    - RPM limited (2500 +/- 500) - 675 files (35% of the original dataset)
        - misalignment, 173, 25.63
        - outer race fault, 137, 20.30
        - cage fault, 136, 20.15
        - imbalance 118, 17.48
        - ball fault 95, 14.07
        - normal, 16, 2.37
    - RPM unlimited
        - misalignment, 498, 25.53%
        - cage fault, 376, 19.27
        - outer race fault, 372, 19.07
        - imbalance, 333, 17.07
        - ball fault, 323, 16.55
        - normal, 49, 2.51
- **Counts of anomaly** (to table) - **Unbalanced dataset**
    - RPM limited
        - anomaly, 0.6 = False 388 True 287
        - anomaly, 0.9 = False 563 True 112
    - RPM unlimited
        - anomaly, 0.6 = False 1125 True 826
        - anomaly, 0.9 = False 1638 True 313
    
- **Correlation of features with rpm**:
    - Temporal: mostly **very low** (25% - 75% quantile: 0.07 - 0.20,
    - Spectral: mostly **very low** for all 5 window sizes: 25% (0.08)  - 75% (0.23)

#### Machinery and measurement
- KALORIK BASIC Stand Fan TKGVT1037C  - height 125 cm, stable 60 cm cross base, 3 speed, 45 W, 45 cm fan diameter, 3 propelers
- (Datacentre SHC3, Petržalka) AC unit VERTIV: Scroll Compressor: Copeland ZR16M3E-TWD-561 (2900 RPM @ 50 Hz; 9.7 kW (13 HP); 380/420V; 25 A) - 2 units
    - Faulty parts: unbalance of the rotor, fault of the scroll, mechanical assembly loose, bearing loose
- (Pumping station Podunajské Biskupice) Water pump: KSB Omega 300-560 B GB G F (2018, 1493 (1500) RPM @ 50 Hz; 400 kW elektromotor; power requirement: 380 kW, Class III - ISO) - 1 unit
    - Faulty parts: misalignment, bearings
- Water pump Sigma Lutín - 1986
- Columns: t, x, y, z, label - worse, better, pump - good, warning, fault (based on ISO)

#### Sensors:
- - Digitálny (SPI), : ADXL335  
    - Axis: 3 axis
    - Analog
    - Noise density: 150 - 300 $\mu g / \sqrt{Hz}$ rms
    - Sensitvity: 300 mV/g
    - Resolution: 12 bits
    - Range: $\pm$ 3g,
    - Sample time: 400 ns (2.5 kHz)
    - Bandwidth: 0.55 kHz
    - Voltage range: up to 1.8 V
    - Microcontroller: Beaglebone Black  (TI Sitara AM3358)

- **Accelerometer**: IIS3DWB (STEVAL-MKI208V1K)
    - Axis: 1 / 3
    - Digitálny (SPI), 
    - Noise density: 75 $\mu g / \sqrt{Hz}$
    - Sensitvity: 0.061 mg/LSB pri 2g
    - Resoltution: 16 bits
    - Range: 2 - 16g
    - ODR: 26.7 kHz
    - Bandwidth: 5 - 6.3 kHz (-3 dB)
    - FIFO: 3 kB (512 vzoriek)
    - Microcontroller:  OLIMEX ESP32-PoE-ISO Rev I (ESP32-WROOM-32)
 
  - **UML diagram**:
      - Block diagram for HW connections:
          - Accel (SPI, INT1,2) -> ESP -> (SDIO) SD card
              - ESP-IDF
              - Detached Switch (INT [34] replace button) and LED (GPIO [5]) to indicate recording
              - **Storage API (SDMMC)** SD card: (GPIO15 [HS2_CMD], GPIO14 [HS2_CLK], GPIO2 [HS2_DATA0])
                  - https://docs.espressif.com/projects/esp-idf/en/v5.1.2/esp32/api-reference/storage/fatfs.html#using-fatfs-with-vfs-and-sd-cards
              - **Peripherals API (SPI Master Driver)** Accel: (3.3V, GND, MISO [16], MOSI [32], CLK [16], CS [13], INT1 [35], INT2 [36]) Sensor max. 10 MHz
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

# TODO: complete with all subsets

### Feature selection

- MaFaulDa features subsets (12 = rpm limit no, 12 = rpm limit yes)
    - Domain: temporal, spectral
    - Classification: fault (multiclass), anomaly (binary - 60%, 90%)
    - RPM limit: no, yes (2500 $\pm$ 500)
  
    - All 6 Fault classes: shaft(normal, imbalance, misalignment) + bearings (cage, ball, outer race)
    - All Placements: A, B

- Validation:
- **Batch**
    - Magnitude of 3D feature vector
    - Balancing dataset with oversampling minority classes
        - **TODO:Dataset sizes in all experiments (adj/non adjust)**
    - Hold-out validation - split to train and test set (80/20)
- **Online**: Order by severity
    - Severity:
        - Number fault severities by sequence
        - Keep only decimal numbers in severity
        - Number severity per group (0 - best, 1 - worst)
        - Transform severities to range (0, 1)
     
- Feature correlations
     
- Best features:
    - Compute metrics: Corr, F stat, MI
    - Compute Rank product and order descending
    - **Majority voting**:
        - Take first feature, remove correlation above 0.95 if correlated feature is already in set
        - **subsets of 2, 3, 4, 5 members?**
        - **Variance non scaled, scaled?**
        - **Variance PCA?**
        - Count in how many sets (**subsets of 3 members**) it is present - choose 3 best
            - Global best (max. score: 24  = experiments):
                - Temporal: std (20), shape (10), skewness (9)
                - Spectral: entropy (14), roll_off (14), noisiness (14)
            - rpm limit (yes/no) and hardware (shaft/bearings) (max score = 6):
                - Temporal:
                    - no, shaft: \{std (4), skewness (3), pp (3)\}
                    - no, bearings: std (4), shape (3), kurtosis (3)
                    - yes, shaft: std (4), skewness(2), pp (3)
                    - yes, bearings: std (6), shape (4), margin (3)
                - Spectral:
                    - no, shaft: std, centroid, roll_off (4)
                    - no, bearings: entropy (4), centroid (3), roll_off (3)
                    - yes, shaft: flux, noisiness, entropy
                    - yes, bearings: energy, entropy, roll off
            - rpm limit (yes/no) and hardware (shaft/bearings) and target (fault, anomaly) (max score = 3):
    - **Rank product**:
        - Apply rank product to all final feature rankings
            - Global best
                - Temporal: std, rms, pp, shape
                - Spectral: entropy, std, roll_off, noisiness

- **TODO: Scores ranges to relative units** (Corr, F stat, MI)
    
### kNN classification
Models-Batch/kNN.html

- Classifiaction with **all extracted features** - min-max scaled, magnitude (knn n=5, euclidian, ) - **target - shaft faults**:
    - Most common misclasiffication between horizontal nad vertical mislignment (up to 8%)
    - Temporal:
          - Train accuracy: 95.91 %
          - Test accuracy: 94.37 %
    - Spectral (2^14):
        - Train accuracy: 97.54 %
        - Test accuracy: 95.80 %
- Classification with globally best features (rank) only for fault

- Permutation models (best accuarcies for given features, lowest error rate)
    - nC3 = models - best accuracy with features
        - 2 tables (temporal, spectral) x 7 rows 
- Accuarcies with chosen sets of features

Models-Online/kNN.html
- Online (Progressive valuation):
    - Gradual learning
        - Fault (shaft) / anomaly
    - Window learning
        - Fault (shaft) / anomaly
        - Compare classification accuracies for window sizes in one graph: (1, 10, 50, 100, 250)
        - Scenarios: fault, anomaly
    - Missing labels
        - Faults
        - Anomaly
    - Scatter plot - True labels vs. Predicted labels
        - Faults
        - Anomaly


### DBSCAN clustering

Models-Batch/DBSCAN.html
- SIlhoutte scores
    - globally best clustering (maximazing silhouette score)?
    - for best feature subsets

 ### Design semi-supervised feature selection
- Feature selection -> KNN -> Infer labels -> Feature selection

- Models-Online/DenStream.html
    - evolution of silhoutte score