1. Zhrnutie obhliadky 
	- analýza datasetov z obhliadky (**BeagleboneEDA.ipynb**)
	- systematické chyby vo vzorkovacej frekvencii
2. Zhrnutie obhliadky - Aké zariadenia merať? Air Conditioner (AC)
	  - **Nemáme tachometer** (otáčkomer) signál!
	- Vysokotlakové **piestové čerpadlo** (EXCOOL) - iba nad 26°C beží 
		- Adiabatic Pumps EVADI  - BC 21/20 S Plunger Pump+ Elektromotor(1430 RPM @ 50 Hz; 3 kW; 400 V; 6,7 A)
		- Faults:
			- Pump Cavitation
			- Bent Pump Shaft
			- Pump Flow Pulsation
			- Pump Impeller Imbalance
			- Pump Bearing Issues
			- Misalignment of the Shaft
	- Kompresor (EXCOOL) - Scroll Compressor - **Špirálový kompresor**
		- Danfoss DSH184A4ALC  (2900 RPM @ 50 Hz; 13 kW;  380/420V; 25 A)
	- **Kompresor (VERTIV) - Scroll Compressor**
		- Copeland ZR16M3E-TWD-561 (2900 RPM @ 50 Hz; 9,7 kW (13 HP); 380/420V; 25 A) 3500 RPM @ 60Hz
		- Faults (https://www.mdpi.com/1099-4300/17/10/7076)
			- unbalance of the rotor, 
			- fault of the scroll
			- mechanical assembly loose
			- bearing loose
4. Zhrnutie obhliadky - Kde merať? (zmysel to má v odľahlých oblastiach)
	- Digitalis (EXCOOL, VERTIV)
	- SHC2 - do konca roka 2023
	- VNET HQ (VERTIV)
5. Merač - potrebujeme niečo bez operačného systému, ideálne vzdialene ovládané, čo vieme spustiť opakovane, ukladá merania vo vzorkách. Spracovanie neskôr
6. Spolupráca so SjF (Žiaran)? - 2 stretnutia: 
	- návrh merania, analýza výsledkov (anotácia poruchy)
	- Nie: iba na záver verfikácia - musí vedieť čo je to presne za stroj inak to nevie povedať
1. Feature extraction - **Feature.ipynb**: Time domain, Frequency domain, Wavelets, PCA, Filtre, EDA 
2. Čo ďalej? - ako určiť potrebné features? - feature selection -
3. Čo ďalej? - ako to dať do modelov - anomaly vs KNN. Aké triedy utvoriť - mám fault (10), severity (5), rpm (600 - 3000 rpm) 
4. Stretnutie - o 3 týždne - 7.týždeň semestra - 30.10.

### Zápisnica
- Merať kompresory a porovnať dobrý zlý , (SHC2 dobrý )
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


**Kompresor (VERTIV) - Scroll Compressor**
		- Copeland ZR16M3E-TWD-561 (2900 RPM @ 50 Hz; 9,7 kW (13 HP); 380/420V; 25 A) 3500 RPM @ 60Hz
		- Faults (https://www.mdpi.com/1099-4300/17/10/7076)
			- unbalance of the rotor, 
			- fault of the scroll
			- mechanical assembly loose
			- bearing loose
##### Scroll Compressor
https://www.youtube.com/watch?v=yNgqI4XPUZc
https://www.youtube.com/watch?v=YNeoFebbU6I
https://www.youtube.com/shorts/nHc4jFTTFl0

![[entropy-17-07076-g001-1024.png]]

##### Reciprocating plunger pump

https://www.youtube.com/watch?v=M9Vt-GpK3_Q
https://www.youtube.com/watch?v=VnPARxLfJ9c

![[bomba_reciprocante_componentes_1.png]]

1. Suction pipes
2. Suction valves in the pump body
3. Plungers
4. Discharge pipes
5. Discharge valves in the pump body
6. Plunger rod
7. Runner
8. Crosshead
9. Crank
10. Sprocket
11. Crankshaft
12. Gear rim

![[Pasted image 20231007230557.png]]