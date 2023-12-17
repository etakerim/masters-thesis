# Design


### Feature extraction

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
D) **Correlations among features**

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
