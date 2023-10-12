
### Tasks
- Merať kompresory a porovnať dobrý zlý , (SHC2 dobrý a zlý, Digitalis dobré, HQ zlé )
- Doplniť do spektrogramov v BeagleBone EDA dB stupnicu
- Implementovať feature selection nad mafaulda a tri modely
- Features
	1. Normovať PCA - minmax scaler
	2. MI - elbow detection
	3. Wavety spraviť MI po layers
- Plán meraní na kompresoroch dobrý a zlý
- Po - namerať klímy v HQ (orientačne)
- Shiratech - v stredu sa zastaviť + programátor
- BeagleBone ADC - RTOS pre BeagleBone

### Knižnice
Time Series Feature Extraction Library (TSFEL for short) is a Python package for feature extraction on time series data.: https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html


**Kompresor (Klíma VERTIV) - Scroll Compressor**
		- Copeland ZR16M3E-TWD-561 (2900 RPM @ 50 Hz; 9,7 kW (13 HP); 380/420V; 25 A) 3500 RPM @ 60Hz
		- Faults (https://www.mdpi.com/1099-4300/17/10/7076)
			- unbalance of the rotor, 
			- fault of the scroll
			- mechanical assembly loose
			- bearing loose
##### Scroll Compressor
https://www.youtube.com/watch?v=yNgqI4XPUZc
https://www.youtube.com/watch?v=YNeoFebbU6I

![[entropy-17-07076-g001-1024.png]]
