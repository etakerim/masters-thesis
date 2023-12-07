# Obhliadka

- Dátum: 3.10.2023, 9:00 - 10:30
- Prítomný: Miroslav Hájek, Marcel Baláž, Lukáš Doubravský, Michal Ország


Pozor (skontrolovať): Výsledky v graphs môžu byť nepresné vo frekvenciách, vzorkovač nemusí byť dobre prerátaný a možno sa líši o násobok (zrýchlený záznam).


## Vybavenie
- Notebook Acer Aspire 5
- Smartfón Motorola Moto
- BeagleBone Black
- Akcelerometer ADXL335
- USB B kábel
- Obojstranná páska (Acrylic double sided tape)
- Nožnice

## Stroje
- (Ventilátor KALORIK BASIC Stand Fan TKGVT1037C  - výška 125 cm, stabilná 60 cm krížová základňa, 3 rýchlosti, 45 W, 45 cm priemer vrtule, 3 vrtule)
- Klimatizácia
	- Sála 1 - VERTIV
	- Sála 3 - EXCOOL
	- SHC2 - VERTIV  (striedavo sa vypínal a zapínal kompresor, to vidieť viacej harmonickými, meranie na vrchu skrine, y smerom dozadu od predného panelu)
- Rotačná UPS
- Šum pri pozadia (na podlahe) pri strojoch

## Otázky
- Čo sa zvykne kaziť?
- Kde má stroj rotačné časti?
- Aké má parametre - výkon, rotačné rýchlosti?
- Aké sú možnosti montáže senzoru čo najbližšie rotačnej časti? - spôsoby pripevnenia - pevný vs. acrylic foam
- Má stroj tachometer?
- Aká je drôtová/bezdrôtová konektivita?
-  Možnosti elektrického napájania (aj tak pôjdeme z batérie, ale pre istotu)?

### Beaglebone ADC
http://beaglebone.cameon.net/home/reading-the-analog-inputs-adc
https://www.123led.sk/lepiaca-paska-hs-alufix/
- Resolution: 12 bits (Output values in the range 0 - 4095)
- Sample time: 125 ns (8 kHz)
- Voltage range: 0 - 1.8 V

## Prieskumné merania
U každého stroja si potrebujeme zmerať základné charakteristiky (v osiach X, Y najcitlivejšie)
- [ ] F - Smartfón (Fotoaparát)
	-  Stroj (Klíma)
	-  Montážne body
	-  Poznámky o stavbe stroja
- [ ] A-SPC - Smartfón (Audio)
	- Aplikácia: Spectroid
	- https://play.google.com/store/apps/details?id=org.intoorbit.spectrum&hl=en
	- Výstup: Screenshoot frekvenčného spektra so špičkami
	- Nastavenia:
		- Source: Microphone
		- Frequency axis scale: Logarithmic
		- Subtract DC: Áno
		- Sampling rate: 48 kHz
		- FFT Size: 4096 (8192) bins
		- Decimation: 5 - 0.18Hz/bin @DC)
		- Window function: Hann
		- Desired transform interval: 50 ms
		- Exponential smoothing factor: 0.5
- [ ]  A-REC Smartfón (Audio)
	- Aplikácia: Voice Recorder (Hlasový záznamník)
	- https://play.google.com/store/apps/details?id=com.media.bestrecorder.audiorecorder&hl=en_US
	- Výstup: Záznam zvuku
	- Nastavenia:
		- Automatic Gain Control (Manual)
		- Mono - 44 kHz - WAV
- [ ] S-SACC Smarfón (Akcelerometer)
	-  Aplikácia: Accelerometer Analyzer
	-  https://play.google.com/store/apps/details?id=com.lul.accelerometer&hl=en_US
	- Výstup: Záznam akcelerácie
	-  Nastavenia:
		- Sensor units: Meters
		- Remove Eartch Gravity: False
		- Sensor speed: Fastest (200 Hz)
- [ ] B-ACC Beaglebone Black
	-  Aplikácia: sampler.c
	-  Výstup: stroj_fs_bod.tsv
	- Nastavenia:
		- Vzorkovacia frekvencia: 4 kHz
		- Umiesnenie senzoru: Krabička vertikálne prilepená obojstrannou lepiacou páskou (fotka)
	- Postup:
```bash
# debian:temppwd

ssh debian@192.168.7.2
gcc sampler.c -o sampler
sudo nice -n -20 ./sampler out.tsv
scp debian@192.168.7.2:/home/debian/out.tsv out.tsv
```

