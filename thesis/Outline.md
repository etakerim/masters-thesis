TODAY:
- images
- tables - bearing frequencies
- graphs
- outline to chapters
- UML

Analysis
+ Rename:
    + Time-domain features
    + Frequency-domain domain

+ Feature extraction -> Bearing fault frequencies
    + Add Bearing fault calculation
+ Check accuracies from cited sources
+ Effect of imbalance on Knn

Design
+ Research questions
+ MaFaulDa EDA
    + Fault counts
    + Correlation to RPM
    + Features Range
    + Time domain waveforms of different faults
    + PCA loading plots
+ MaFaulDa pipeline UML: Split to parts -> Remove mean -> Low pass -> (TD features / Spectral transform -> FD features) -> MinMax Scaler -> Labeling -> Online learning // Balancing -> (All features / Choose feature comination / PCA) KFold -> k-NN -> Accuracy
+ Sensor device description
    + HW parameters
    + Block diagram
    + Firmware UML


Implementation
+ Jupyter notebooks - screenshot
+ HW build (image of device)
+ Firmware in C (timing too slow - osciloscop) and bin2csv convertor

+ Machinery description + historgrams of vibrations levels + features ranges
    + Fan
    + Compressor
    + KSB Pump + Bearings
    + Sigma pump
    + KSB Cloud

+ Measuremnt UML - Experiment step-by-step
    + Measurement points and orientation
    + Measuremnt plan (calendar)
    + UML: Loop{Tape sensor to chosen point, Turn On device, Download from SD card, Convert from bin to csv}->Zip to Dataset
        -> pipeline (extract features in time and frequency domain) ->

-------------------------------------------------------------------------------------------------------------------------

Evaluation
+ MaFaulda
    + All features
    + Feature cominations (box plots of distributions)
    + Feature selection methods
        + table of accuracies
        + compare dimensions
        + compare percentiles
        + 2d and 3d plots of best features
    + Online learning - tumbing windows and label skips
+ Fan - estimating rotation speed (audio, high speed camera, custom fw validation)
+ Pumps and compressors
    + frequency spectrum
    + Bearing fault analysis
    + Time-frequency spectrum (machines, turn-on, turn-off, interference from nearby machine)
+ KSB cloud dataset (BVS analysis)


Conclusion
+ Comprehensive study of monitoring vibrations and checking it with MaFaulDa dataset
+ All research questions answered
+ Project work guided thourgh risk management in collaboration with partners
+ Main benefits - Data processing of vibartions in industrial low-powered IoT devices


Instalation manual
- Tested on Manjaro Linux with KDE plasma Linux Acer 6.1.85-1-MANJARO x86_64 GNU/Linux
- pip install
- how to flash firmware

User manual
- run notebooks


--------------------------------------------------------------------------------------------
Current
3 Design
 39
3.1
 Research questions . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
3.2
 Dataset exploration . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
3.2.1
 Fault annotations . . . . . . . . . . . . . . . . . . . . . . . . . 42
3.2.2
 Signal filters . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
3.2.3
 Statistical tests . . . . . . . . . . . . . . . . . . . . . . . . . . 44
3.3
 Feature relevance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
3.3.1
 Feature extraction . . . . . . . . . . . . . . . . . . . . . . . . 44
3.3.2
 Data volume savings . . . . . . . . . . . . . . . . . . . . . . . 47
3.3.3
 Feature selection . . . . . . . . . . . . . . . . . . . . . . . . . 49
3.4
 K-nearest neighbor classifier . . . . . . . . . . . . . . . . . . . . . . . 50
3.4.1
 Batch models . . . . . . . . . . . . . . . . . . . . . . . . . . . 51
3.4.2
 Online models . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
4 Implementation
 61
4.1
 Machinery for monitoring . . . . . . . . . . . . . . . . . . . . . . . . 61
4.2
 Sensor hardware and drivers . . . . . . . . . . . . . . . . . . . . . . . 63
4.3
 Firmware . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
4.4
 Dataset description . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
4.5
 BVS cloud . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
4.6
 Compressors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
4.7
 Water pumps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
5 Evaluation
 67
6 Conclusion
Bibliography
69


