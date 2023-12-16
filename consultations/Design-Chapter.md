# Design

The design phase sets out to elaborate on the scenarios for experiments with IoT device. In these experiments, we compare and combine approaches from industry standards and papers presented in the analysis section. Noted observations are taken into consideration in establishing the plan of measurements and in the construction of the sensor unit. Finally, the network infrastructure layout is provided where we propose optimizations in the functions of the components.

## Research questions

This thesis aims to provide answers to four broader formulated research questions. The focus is primarily on making data flow more efficient in an industrial sensor network that monitors rotating machines. The research questions are:
1. Which temporal and spectral features can be extracted from vibration signals to provide the most accurate record of machinery faults?
2. What is the reduction in transmission goodput when chosen signal features are used?
3. What accuracies of prediction models can be achieved with various feature subsets? 
4. How can machinery faults be continuously identified and predicted based upon collected events?

In accomplishing the objectives of our research we propose several goals:
- Statistically and visually describe vibration signals from the Machinery fault database (MauFaulDa).
- Establish a list of conditions that should be later investigated in the experimental setting.
- Prepare dataset to be used in conjunction with statistical learning models, namely by identifying labels and balancing classes.
- Find the best subsets of features in temporal and spectral domain with previously analyzed feature extraction and selection methods.
- Evaluate the performance of models described in the diagnostics section with a significant focus on the k-nearest neighbor algorithm.
- Acquire measurements of vibrations from machines in the real environment to form a novel dataset of machinery behavior.
- Develop hardware and implement its firmware to obtain such measurement in the quality demanded by vibrodiagnostics standards.
However, we leave out from our efforts experiments on the data features from wavelets and on the peaks in the spectral domain. The reason is that we did not find a way to represent extracted features more succinctly as a one number. We also did not discover a strategy for choosing only the relevant frequency bins. The assembled description is kept to empower further research on that topic.

The tasks are associated with risks impacting their successful completion. The possible risks are assessed and prioritized. The most notable risk encountered early on is that the machine in the real environment will not be available for vibration measurements. We successfully eliminated the risk by contacting and establishing collaboration with alternative partners. The additional risks are that the repeated measurements will not be consistent and fault modes could not be reliably differentiated and labeled in the dataset because each machine is unique in its structure. Related is the risk that not enough data is obtained from various classes.  Also, suggestions can made by exploring the MaFaulDa dataset which will not be applicable in the practice. All these risks have to be regularly reevaluated to achieve our goals.
  
###  Dataset exploration

#### TODO -----------------------
robíme so skôr popísanou mafaulda ktorú musíme dostať do stavu vhodné na prieskum
vybrali sme si tento dataset pretože je narozsiahlejší, čím umožňuje vybať podmnožiny zamernaé na špecifické situácie. Má s viacerými označenými typmi.
najprv si vyberieme vzorové záznamy z každej porchy jeden a až neskôr extrahujeme atribúty z celej množiny
zaujíma nás vizualizovať časový a frekvenčný priebeh
a tiež štatistické vlastnosti rozdelenia v oboch doménach

Ako sme popísali v analýze mafaulda dataset sa skladá z 1951 záznamov olabelovaných simulovanou poruchou stúpajúcej závažnosti.
Záznam obsahuje časové rady so vzorkovacou frekvenciou 50 kHz z trojosových akcelerometrov (x, y, z) umiestnené na dvoch ložiskách. 
Inner bearing closer to motor sa označuje aj ako underhang (A), outer bearing sa označuje aj ako overhang (B). 

- Preprocessing

Z 10 labelov pre poruchy - z toho 1 baseline,  3 pre hriadeľ , 3 pre inner bearing a 3 pre outer bearing vytvoríme 6 tried pre neskoršiu klasifikáciu
Pri chybách hrideľa nebudeme rozlišovať smer pri misalignment - zlúčime horizonal and vertical under one category. 
Podľa pozície bearing vyfiltrujeme (choose) poruchy, ktoré sa ho týkajú. čiže ak sa vyskytuje určitá porucha napr. inner bearing nebudeme pri druhom bearing brať, že signál z neho postihuje táto porucha.
To nám zanecháva 6 typov identifikovaných porúch - 1 baseline, 2 pre shaft (imbalance, misalignment) a 3 pre bearing (cage fault, ball fault, outer race fault). Uvažovali sme aj s rozdelením na poruchy shaft a fault, ale v realite sa nedajú oddeliť. Ale napr. typ poruchy ložiska sa môže prejaviť na hriadeli a naopak.
V kroku výberu ložiska dosiahneme to, že odstránime niektoré záznamy, ktoré sa týkajú druhého ložiska a pri premenovaní typov porúch dochádza k ich zlúčeniu tak ako sme popísali a úpravou názvov idenfikátorov získaných z názov súborov

Závažnosť poruchy pre rozličné typy porúch určíme tak, že masses attached a shifting the motor shaft zoradíme in ascedning order and order number will indicate severity. Aby sme mohli pracovať rovnako s poruchami pre ktoré bolo realizovaný rôzny počet podmienok we scale poradia do intervalu from lowest severity 0 to 1 highest severity.

V datasete sú zaznamenané rôzne rýchlosti otáčok, ktoré majú vplyv tak silu prejavu poruchy. RPM najprv vypočítame zo stĺpca tachometera, kde je rýchlosť zaznamenaná ako pwm impulzy. Zistíme rozdiely medzi nasledujúci nábežným hranami a priemer intervalov po prevode na otáčky minúty bude predstavovať údaj o rotačnej rýchlosti (60 / delta t).

Exemplár pre každú poruchy zvolíme podľa závažnosti poruchy a rýchlosti otáčok. Aby sme pri porovnávaní exemplárnych priebehov znížili efekt rozdielov rotačnej rýchlosti zvolíme takú ktorá je blízka priemeru zaznamených rýchlostí a vyššia najbližšia 2500 rpm (42 Hz). Aby sme videli najsilnejší prejav poruchy zvolíme najvyššiu severitu z každej poruchy okrem nomálneho stavu, kde je severita len jedna a síce najnižšia.

Vibračný signál v troch smeroch upravíme najprv odstránením jednosmernej zložky odčítaním celkového priemeru. Potom použijeme ešte digital IIR Butterworth low pass filterom of 5th order with -3dB cutoff frequency 10 kHz. Všimli sme si špičky na 20 kHz, ktorá nemohla byť spoľahlivo zaznamená kvôli lineránej frequency response senzora do 10 kHz a zároveň je to outside of range any commerial mems accelerometer.

Časové priebehy 300 ms segmentu (výrezu) signálu, ktorý začína v čase 1 sekunda pre každý exemplár poruchy sú znázornené v grafoch na obr. X. Z prava doľava sú v stĺpcoch jednotlivé smery radial, tagentail and axial. Amplitudes range from $\pm 3\;m /s^2$** in normal and misaligned data to  **$\pm 11\;m /s^2$** in case of severe bearing faults. In frekevenčné spektrum na obr. Y bolo vypočítané cez FFT with Hann window of length $2^{14}$, čo predstavuje približne 328 ms úsek a spektrálne rozlíšenie zhruba 3 Hz. Graf bol orezený v oboch osiach aby sa dali pozorovať najdôležitejšie črty. Preto ide rozsah frekvencií na vodorovnej osi iba do 3 kHz.

- Statistical tests
Na vzorových poruchách sme uskutočnili štatistické testy, ktoré majú prezradiť charkter vstupných signálov pre ďalšie kroky spracovania. Testami a vizuálne sme overovali normálnosť rozdelenia a statcionaritu dát na segmente o trvaní pol sekundy, ktorý downsample-neme faktorom 10, aby obsahoval 2500 pozorovaní amplitúdy.

  Ako significance test for noramlity použijeme Shapiro-Wilk's test, ktorý in tempoal domain s p-value 0.05 zamieta vo väčšine prípadov hypotézu že data is drawn from normal distribution. In spectral domain up to 3 kHz the null hypotesis is in all cases rejected with p-value less than 0.001. Axis not exhibing the fault have noraml distribution while does where fault is strongly pronounced does not. Null hypotesis cannot be rejected when using lower severity level because the signal is closer to white (pink) noise. In quantile–quantile plot we observe jednoznačný tilt to the diagonal line.

  Augmented Dickey-Fuller test rejects the null hypothesis of unit root with p-value less than 0.001 signifying stochastic process being stationary in nature. This is confirmed with shape of autocorrelation plots.




- **Feature extraction**

A) **Activity diagram** (Image)
1. Label recording using filename path in zip: fault, severity, seq
2. Split each recording to 5 parts (1 second - 50000 samples)
3. Calculate rpm (out of impulse tachometer = prominence=3, width=50, 60 / $\Delta t$)
4. Remove mean - DC component filter
5. Filter frequencies: cutoff = 10 kHz (Butterworth lowpass filter of 5 order) - peak 20 kHz (not possibe to capture)
6. Label with 6 classes of faults, and 2 anomaly classes (based on arbitrary severity 0.6 and 0.9)
7. X - extract temporal features in all axis for both bearings A, B: ax/ay/az_feature_name
8. X - window size: windows and welch averaging (2**14 = 16384 samples, 3 okná, 3.05 Hz resolution)
B) **Counts of classes** (to table and Sum) - **Unbalanced dataset**  (2 Tables - `RPM unlimited | RPM limited`)
C) **Correlation of features with rpm**:
    - Temporal: mostly **very low** (25% - 75% quantile: 0.07 - 0.20,
    - Spectral: mostly **very low** for all 5 window sizes: 25% (0.08)  - 75% (0.23)


- Feature selection

### MaFaulDa predictions


### Machinery 

### Accelerometer sensors

### Preliminary measurements




----
# Design in Masters Thesis


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
        - Visualizations (A, B separately - **we are intersted in closest bearing**)
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
- **Water pump**: KSB Omega 300-560 B GB G F
    - (2018,
    - 1493 (1500) RPM @ 50 Hz;
    - 400 kW elektromotor;
    - power requirement: 380 kW,
    - Class II - ISO) - 1 unit
    - Pumping station Podunajské Biskupice
    - Faulty parts: misalignment, bearings
- **Water pump** Sigma Lutín - 1986
- Columns: t, x, y, z, label - worse, better, pump - good, warning, fault (based on ISO)

#### Sensors:

| Accelerometer                           | ADXL335          | IIS3DWB            |
|-----------------------------------------|------------------|--------------------|
| Vendor                                  | Analog Devices   | STMicroelectronics |
| Bus                                     | Analog           | SPI                |
| Axis                                    | 3                | 1 or 3             |
| Range (g)                               | $\pm$ 3          | $\pm$ 2 to 16      |
| Bandwidth (Hz)                          | 550              | 5 - 6.3            |
| Sensitivity                             | 300 mV/g         | 0.061 mg/LSB       |
| Noise density ($\mu g / \sqrt{Hz}$ rms) | 150 - 300        | 75                 |
| Microcontroller                         | Beaglebone Black | ESP32-PoE-ISO      |
| CPU SoC                                 | TI Sitara AM3358 | ESP32-WROOM-32     |
| Output data rate (kHz)                  | 2.5              | 26.7               |
| A/D resolution (bit)                    | 12               | 16                 |
| FIFO                                    | -                | 3 kB (512 samples) |
 
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
   
#### MaFaulDa feature extraction:
1. Label recording using filename path in zip: fault, severity, seq
2. Split each recording to 5 parts (1 second - 50000 samples)
3. Calculate rpm (out of impulse tachometer = prominence=3, width=50, 60 / $\Delta t$)
4. Remove mean - DC component filter
5. Filter frequencies: cutoff = 10 kHz (Butterworth lowpass filter of 5 order) - peak 20 kHz (not possibe to capture)
6. extract temporal features in all axis for both bearings A, B: ax/ay/az_feature_name
7. spectral 5 window sizes: windows and welch averaging (2**14 = 16384 samples, 3 okná, 3.05 Hz resolution)
8. Label with 6 classes of faults, and 2 anomaly classes (based on arbitrary severity 0.6 and 0.9)
- TBD: Additional features:
    -**Find harmonics** (peaks with MMS and filter with threshold mean+2*std) and harmonic series
    -**WPD energy**, kurtosis

- Fault classes:
    - 'normal': 'normal',
    - 'imbalance': 'imbalance',
    - 'horizontal-misalignment': 'misalignment',
    - 'vertical-misalignment': 'misalignment',
  A
    - 'underhang-cage\_fault': 'cage fault',
    - 'underhang-ball\_fault': 'ball fault',
    - 'underhang-outer\_race': 'outer race fault'
B
    - 'overhang-cage\_fault': 'cage fault',
    -  'overhang-ball\_fault': 'ball fault',
    - 'overhang-outer\_race': 'outer race fault'
 
**Counts of classes** (to table and Sum) - **Unbalanced dataset** - ClassCounts.ipynb
    - RPM limited (2500 +/- 500)

|            fault | A_rpm_nolimit | A_rpm_nolimit_percent | A_rpm_limit | A_rpm_limit_percent | B_rpm_nolimit | B_rpm_nolimit_percent | B_rpm_limit | B_rpm_limit_percent |
|-----------------:|--------------:|----------------------:|------------:|--------------------:|--------------:|----------------------:|------------:|--------------------:|
|   misalignment   | 498           | 34.631433             | 173         | 34.325397           | 498           | 35.750179             | 173         | 36.192469           |
|     imbalance    | 333           | 23.157163             | 118         | 23.412698           | 333           | 23.905240             | 118         | 24.686192           |
|    cage fault    | 188           | 13.073713             | 67          | 13.293651           | 188           | 13.496052             | 69          | 14.435146           |
|    ball fault    | 186           | 12.934631             | 61          | 12.103175           | 137           | 9.834889              | 34          | 7.112971            |
| outer race fault | 184           | 12.795549             | 69          | 13.690476           | 188           | 13.496052             | 68          | 14.225941           |
|      normal      | 49            | 3.407510              | 16          | 3.174603            | 49            | 3.517588              | 16          | 3.347280            |

**Counts of anomaly** (to table) - **Unbalanced dataset**
0.6
| A_rpm_nolimit | A_rpm_nolimit_percent | A_rpm_limit | A_rpm_limit_percent | B_rpm_nolimit | B_rpm_nolimit_percent | B_rpm_limit | B_rpm_limit_percent |           |
|--------------:|----------------------:|------------:|--------------------:|--------------:|----------------------:|------------:|--------------------:|----------:|
|         False |                   837 |   58.205841 |                 375 |     74.404762 |                   831 |    59.65542 |                 354 | 74.058577 |
|          True |                   601 |   41.794159 |                 129 |     25.595238 |                   562 |    40.34458 |                 124 | 25.941423 |

0.9
| A_rpm_nolimit | A_rpm_nolimit_percent | A_rpm_limit | A_rpm_limit_percent | B_rpm_nolimit | B_rpm_nolimit_percent | B_rpm_limit | B_rpm_limit_percent |           |
|--------------:|----------------------:|------------:|--------------------:|--------------:|----------------------:|------------:|--------------------:|----------:|
|         False |                  1227 |   85.326843 |                 375 |     74.404762 |                  1197 |   85.929648 |                 354 | 74.058577 |
|          True |                   211 |   14.673157 |                 129 |     25.595238 |                   196 |   14.070352 |                 124 | 25.941423 |

- **Correlation of features with rpm**:
    - Temporal: mostly **very low** (25% - 75% quantile: 0.07 - 0.20,
    - Spectral: mostly **very low** for all 5 window sizes: 25% (0.08)  - 75% (0.23)



### Feature selection

- Compressor best:
    - Temporal: margin/impulse/crest, kurtosis, rms
    - Spectral: entropy, centroid, noisiness
    
- MaFaulDa features subsets (12 = online, 12 = batch experiments, **Use decision tree ilustation with shorten leaves**
    - Placement: A, B   - filter only bearing fault for measured bearing, vm and hm to misalign,
    - Domain: temporal, spectral
    - Online: no, yes
    - RPM limit: no, yes (2500 $\pm$ 500)
    - Classification: fault (multiclass), anomaly (binary - 60%), anomaly( binary - 90%)
   
- Validation:
- **Batch**
    - Magnitude of 3D feature vector
    - Balancing dataset with oversampling minority classes
        - **TODO:Dataset sizes in all experiments (adj/non adjust)**
    - Hold-out validation - split to train and test set (80/20)
- **Online**: Order by increaing severity, shuffle samples within severity level
    - Severity:
        - Number fault severities by sequence
        - Keep only decimal numbers in severity
        - Number severity per group (0 - best, 1 - worst)
        - Transform severities to range (0, 1)
     
- Feature correlations (> 0.95):
    - Temporal:
        - std, rms: 1.0
        - std, pp, rms, 0.96
        - pp, max, 0.98
        - crest, margin, implulse, 0.99
    - Spectral:
        - skewness, kurtosis, 0.98
     
- Best features:
    - Compute metrics: Corr, F stat, MI
    - Compute Rank product and order descending
      
- **Majority voting**:
    - Take first feature, remove correlation above 0.95 if correlated feature is already in set*
      - Count in how many sets (**subsets of 3 members**) it is present - choose 3 best
          - Global best = **Rank product** (max. score: 24  = experiments):
              - Batch (12)
                  - Temporal: shape (9, 75%), std (6, 50%), margin (5, 42%), next 3
                  - Spectral: entropy (8, 66%), centroid (6, 50%), std (6, 50%), next flux
              - Online (12) - should be the same at the end
                  - Temporal: shape (10, 83%), margin (9, 75%), std (6, 50%), next 3
                  - Spectral: entropy (9, 75%), flux (7, 58%), centroid (6, 50%), std (6, 50%)
          - Global best = **Correlation** (max. score: 24  = experiments):
              - Batch (12)
                  - Temporal: shape (8), std (6), margin (6)
                  - Spectral: flux (9), entropy (5), std (5)
              - Online (12) - should be the same at the end
                  - Temporal: skewness (11), kurtosis (10), pp (4)
                  - Spectral: roll_on (10), energy (8), roll_off (6)
            - Global best = **F statistic** (max. score: 24  = experiments):
              - Batch (12)
                  - Temporal: shape (8), std (6), margin (6)
                  - Spectral: flux (8), entropy (8), std (5)
              - Online (12) - should be the same at the end
                  - Temporal: skewness (11), kurtosis (10), pp (4)
                  - Spectral: roll_on (10), energy (8), roll_off (7)
            - Global best = **Mutual informtion** (max. score: 24  = experiments):
              - Batch (12)
                  - Temporal: temporal\_std 11 temporal\_shape 10 temporal\_kurtosis 4
                  - Spectral: spectral\_std 10 spectral\_energy 10 spectral\_roll\_off 6
              - Online (12) - should be the same at the end
                  - Temporal: temporal\_kurtosis 9 temporal\_skewness 8 temporal\_crest 7
                  - Spectral: spectral\_energy 12 spectral\_std 6 spectral\_roll\_on 6
        - **Rank product**:
            - Apply rank product to all final feature rankings
                - Global best
                    - Batch (12)
                        - Temporal: std, rms, shape, pp
                        - Spectral: entropy, std, centroid, flux
                    - Online (12)
                        - Temporal: std, rms, shape, pp
                        - Spectral: entropy, flux, std, centoid
                    
Tables to appendix (Rank product, Corr, Rank product, MI)
  
| placement | online | rpm_limit |     target |                  temporal |                         spectral |
|----------:|-------:|----------:|-----------:|--------------------------:|---------------------------------:|
|         A |  False |     False | anomaly_60 |      [margin, shape, std] |        [centroid, entropy, flux] |
|           |        |           | anomaly_90 |         [max, shape, std] |        [centroid, entropy, flux] |
|           |        |           |      fault |         [max, shape, std] |       [flux, roll_off, skewness] |
|           |        |      True | anomaly_60 |      [margin, shape, std] |         [centroid, entropy, std] |
|           |        |           | anomaly_90 |      [margin, shape, std] |         [centroid, entropy, std] |
|           |        |           |      fault |       [margin, pp, shape] |    [entropy, roll_off, skewness] |
|           |   True |     False | anomaly_60 |      [margin, shape, std] |        [centroid, entropy, flux] |
|           |        |           | anomaly_90 |      [margin, rms, shape] |        [centroid, entropy, flux] |
|           |        |           |      fault |         [max, shape, std] | [centroid, negentropy, roll_off] |
|           |        |      True | anomaly_60 |      [margin, shape, std] |             [entropy, flux, std] |
|           |        |           | anomaly_90 |      [margin, shape, std] |             [entropy, flux, std] |
|           |        |           |      fault |       [margin, pp, shape] |   [centroid, roll_off, skewness] |
|         B |  False |     False | anomaly_60 | [kurtosis, rms, skewness] |             [entropy, flux, std] |
|           |        |           | anomaly_90 |       [crest, shape, std] |        [entropy, noisiness, std] |
|           |        |           |      fault |     [crest, pp, skewness] |    [centroid, roll_off, roll_on] |
|           |        |      True | anomaly_60 |    [kurtosis, rms, shape] |           [energy, entropy, std] |
|           |        |           | anomaly_90 |      [margin, rms, shape] |         [energy, noisiness, std] |
|           |        |           |      fault |     [crest, pp, skewness] |   [centroid, noisiness, roll_on] |
|           |   True |     False | anomaly_60 |   [kurtosis, margin, rms] |             [entropy, flux, std] |
|           |        |           | anomaly_90 |         [max, shape, std] |        [entropy, noisiness, std] |
|           |        |           |      fault |     [crest, kurtosis, pp] |        [centroid, flux, roll_on] |
|           |        |      True | anomaly_60 | [kurtosis, margin, shape] |             [entropy, flux, std] |
|           |        |           | anomaly_90 |      [margin, shape, std] |        [entropy, noisiness, std] |
|           |        |           |      fault |       [margin, pp, shape] |     [centroid, entropy, roll_on] |
           
              
             
  | placement | online | rpm_limit |     target |                      temporal |                         spectral |
|----------:|-------:|----------:|-----------:|------------------------------:|---------------------------------:|
|         A |  False |     False | anomaly_60 |   [crest, kurtosis, skewness] |  [kurtosis, negentropy, roll_on] |
|           |        |           | anomaly_90 |   [crest, kurtosis, skewness] |    [kurtosis, roll_on, skewness] |
|           |        |           |      fault | [impulse, kurtosis, skewness] |      [centroid, energy, roll_on] |
|           |        |      True | anomaly_60 |     [kurtosis, max, skewness] |  [negentropy, roll_off, roll_on] |
|           |        |           | anomaly_90 |     [kurtosis, max, skewness] |    [kurtosis, roll_off, roll_on] |
|           |        |           |      fault |   [crest, kurtosis, skewness] |     [energy, noisiness, roll_on] |
|           |   True |     False | anomaly_60 |        [crest, kurtosis, max] |           [energy, roll_on, std] |
|           |        |           | anomaly_90 |   [crest, kurtosis, skewness] |      [energy, roll_off, roll_on] |
|           |        |           |      fault | [impulse, kurtosis, skewness] |           [energy, roll_on, std] |
|           |        |      True | anomaly_60 |     [kurtosis, max, skewness] |      [energy, roll_off, roll_on] |
|           |        |           | anomaly_90 |           [kurtosis, max, pp] |    [kurtosis, roll_off, roll_on] |
|           |        |           |      fault |   [crest, kurtosis, skewness] |           [energy, roll_on, std] |
|         B |  False |     False | anomaly_60 |          [impulse, pp, shape] |  [kurtosis, negentropy, roll_on] |
|           |        |           | anomaly_90 |      [kurtosis, pp, skewness] |  [negentropy, roll_off, roll_on] |
|           |        |           |      fault |      [impulse, margin, shape] | [kurtosis, negentropy, skewness] |
|           |        |      True | anomaly_60 |           [max, pp, skewness] |  [kurtosis, negentropy, roll_on] |
|           |        |           | anomaly_90 |      [kurtosis, pp, skewness] |    [kurtosis, roll_off, roll_on] |
|           |        |           |      fault |     [kurtosis, margin, shape] | [kurtosis, negentropy, skewness] |
|           |   True |     False | anomaly_60 |            [impulse, max, pp] |    [energy, negentropy, roll_on] |
|           |        |           | anomaly_90 |   [impulse, margin, skewness] |  [negentropy, roll_off, roll_on] |
|           |        |           |      fault |   [kurtosis, shape, skewness] |     [energy, kurtosis, skewness] |
|           |        |      True | anomaly_60 |              [crest, max, pp] |  [centroid, negentropy, roll_on] |
|           |        |           | anomaly_90 |      [kurtosis, pp, skewness] |    [centroid, roll_off, roll_on] |
|           |        |           |      fault |   [kurtosis, shape, skewness] |   [energy, negentropy, roll_off] |

  - **PCA explained variance**:?
      - All features in both domains explained by 3 PC components `plot\_explained\_variances`
  - **Silhouette scores**:
      - Ako sú rozlíšiteľné clustre cez najlepšie : `plot\_silhouette\_scores`


    
### kNN classification

#### Models-Batch/kNN.html

- **Classification with all extracted features**** - **min-max scaled**, magnitude **(knn n=5, euclidian metric**)
    - Precision and recall (with micro averaging) were same as accuracy
    - Our aim is to set up baseline for comparison with reduced number of features and online learning
    - row = {'rpm\_limit': False, 'target': 'fault', 'placement': 'A', 'online': False}
    - B has worst precisions
    - In all cases all spectral features have better prection metrics than all temporal features - high correlations and dependency
    - **Anomaly (0.9)** - around 7 times more false positives (1st degree error) than false negatives (for both A and B)
  
|           |                                     | Temporal domain |               |                | Spectral domain |               |                |
|-----------|-------------------------------------|-----------------|---------------|----------------|-----------------|---------------|----------------|
|           |                                     | Train accuracy  | Test accuracy | Train macro F1 | Train accuracy  | Test accuracy | Train macro F1 |
| A bearing | Fault (\sum 2998, 498 pre class)    | 94.13           | 90.93         | 0.91           | 99.17           | 98.36         | 0.98           |
|           | Anomaly (\sum 2454, 1227 pre class) | 96.68           | 95.35         | 0.96           | 99.21           | 98.74         | 0.99           |
| B bearing | Fault (\sum 2998, 498 pre class)    | 86.34           | 79.35         | 0.79           | 92.07           | 87.65         | 0.87           |
|           | Anomaly (\sum 2394, 1197 pre class) | 91.48           | 86.68         | 0.87           | 94.28           | 91.56         | 0.92           |


- **Permutation models** (best accuarcies for given features, lowest error rate)
    - nC3 = models - best accuracy with features = 120 temporal models (10 feat) and 165 spectral (11 feat) to determine best accuracies for 3 features
        - 2 tables (temporal, spectral) x 7 rows
    - **Best accuracies  (RPM unlimited)**
|     target | placement |   domain |                             features | train_accuracy | test_accuracy |
|-----------:|----------:|---------:|-------------------------------------:|---------------:|--------------:|
| anomaly_60 |         A | temporal |         ['skewness', 'rms', 'shape'] |       0.900239 |      0.845281 |
| anomaly_60 |         A | spectral |  ['centroid', 'roll_off', 'entropy'] |       0.940711 |      0.899642 |
| anomaly_90 |         A | temporal |         ['std', 'skewness', 'shape'] |       0.955888 |      0.932763 |
| anomaly_90 |         A | spectral | ['centroid', 'noisiness', 'entropy'] |       0.981764 |      0.970660 |
|      fault |         A | temporal |         ['std', 'skewness', 'shape'] |       0.910559 |      0.859103 |
|      fault |         A | spectral |  ['centroid', 'kurtosis', 'entropy'] |       0.974983 |      0.950469 |
| anomaly_60 |         B | temporal |               ['rms', 'pp', 'shape'] |       0.806558 |      0.694344 |
| anomaly_60 |         B | spectral |     ['centroid', 'std', 'noisiness'] |       0.861613 |      0.793622 |
| anomaly_90 |         B | temporal |            ['std', 'kurtosis', 'pp'] |       0.908521 |      0.855054 |
| anomaly_90 |         B | spectral |       ['centroid', 'std', 'roll_on'] |       0.949457 |      0.918546 |
|      fault |         B | temporal |      ['std', 'skewness', 'kurtosis'] |       0.836931 |      0.765395 |
|      fault |         B | spectral |      ['centroid', 'std', 'roll_off'] |       0.906710 |      0.858434 |

- **Accuracies with chosen sets of features** (Tables)
1. Rank product
|     target | placement |   domain |                       features | train_accuracy | test_accuracy |
|-----------:|----------:|---------:|-------------------------------:|---------------:|--------------:|
| anomaly_60 |         A | temporal |           [std, margin, shape] |       0.867682 |      0.807049 |
| anomaly_60 |         B | temporal |      [kurtosis, rms, skewness] |       0.802948 |      0.693742 |
| anomaly_90 |         A | temporal |              [shape, std, rms] |       0.933272 |      0.891606 |
| anomaly_90 |         B | temporal |            [std, shape, crest] |       0.896094 |      0.850459 |
|      fault |         A | temporal |              [std, shape, max] |       0.904786 |      0.854418 |
|      fault |         B | temporal |          [pp, crest, skewness] |       0.819361 |      0.737952 |
| anomaly_60 |         A | spectral |      [centroid, flux, entropy] |       0.903973 |      0.839307 |
| anomaly_60 |         B | spectral |           [std, flux, entropy] |       0.792268 |      0.693742 |
| anomaly_90 |         A | spectral |      [centroid, flux, entropy] |       0.965363 |      0.947433 |
| anomaly_90 |         B | spectral |      [std, noisiness, entropy] |       0.926483 |      0.884294 |
|      fault |         A | spectral | [roll_off, centroid, skewness] |       0.950385 |      0.921017 |
|      fault |         B | spectral |  [centroid, roll_on, roll_off] |       0.891399 |      0.839023 |

2. Correlation
  ---||---
3. F statistic
---||---
4. MI
   ---||—
   
- * (2 Boxplots) Possible models range of accuracies with given features (3 subplots by target: fault, anomaly60, anomaly_90)
    - **Boxplot** (errorbars): `knn\_train\_accuracy\_range`
        - Train accuracy of all kNN model obtained by combinations of 3 features (unlimited RPM)
        - Green dot is best performance with all features (order according to train accuracy)
    - **Boxplot** (errorbars): `knn\_test\_accuracy\_range`
      
    -  (4 Bar charts) **Scores side by side (bar chart)**
        - all features, pca of all features (n=3), best permuation of 3 feat, rank product (n=3), corr (n=3), fstat (n=3), mi (n=3)
        - Summary
            - Best combinations of 3 features is better than all 10 features because of overfitting
            - PCA (n=3) is always better than any other method of feature selection
            - Rank product is not always better but balances other methods to achieve more stable results. In some situations it it better than all three individually
            - Subset of spectral features have better performance than temporal features, presumbly because of many correlated pairs of features
            
            - *Defining k can be a balancing act as different values can lead to overfitting or underfitting. Lower values of k can have high variance, but low bias, and larger values of k may lead to high bias and lower variance. The choice of k will largely depend on the input data as data with more outliers or noise will likely perform better with higher values of k* (https://www.ibm.com/topics/knn)

        - **knn is lazy learner (no learning time)**
        - RPM unlimited
            - Train
            - Test
        -  - *RPM limited (Later)* - Train, Test

      
    - In future **Investigate** how many features is optimal (gives better better performance) - **altering feature selection procedure/scores**
        - without using PCA because we want to point to relavant indicators behind decision
          
- Relatioship between number of k neighbors and error rate of model (features in model are 3 from best subset)
    - Best is least number of neighbors (3)
    - Temporal
        - **Plot** (Train)
        - **Plot** (Test)
     - Spectral
        - **Plot** (Train)
        - **Plot** (Test)

- **Scatter plots**:
    - **Fault Plot (2 examples in total)** - there is significant overlap in classes
        -  temporal, spectral
    - **Anomaly Plot (2 examples in total - highest accuracy)** - there is significant overlap in classes
        - temporal, spectral

#### Models-Online/kNN.html
- Temporal and Spectral (2 exports)
    - config = {'rpm_limit': False, 'placement': 'A', 'domain': DOMAIN, 'target': 'fault'}
    - config = {'rpm_limit': False, 'placement': 'A', 'domain': DOMAIN, 'target': 'anomaly'}
    - When examples for all classes in labels exists - accuracies of model stabilizes quickly
    - When new label is introduced accuracy drops momentarily
    - Delay, Skiping labels -  degrades model performance
- Online (Progressive valuation):
    - Plot: label ordering in train (in future create strategy for balancing of classes)
        - Faults (Plot)
        - Anomaly (Plot)
    - Gradual learning
        - Fault (shaft) / anomaly (2 plots) - comment on convergence
    - Window learning
        - Fault (shaft) / anomaly
        - Compare classification accuracies for window sizes in one graph: (1, 10, 50, 100, 250)
        - Scenarios: fault, anomaly
    - Missing labels
        - Faults
        - Anomaly
    - Scatter plot - True labels vs. Predicted labels (it is not the case that areas with one fault are homogenous and compact in PC dimensions)
        - Faults 
        - Anomaly
    - **One physical defect can have multiple manifestations - outer race fault and misalignment overlap**


 ### **Design semi-supervised feature selection**
- Feature selection -> KNN -> Infer labels -> Feature selection
