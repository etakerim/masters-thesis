# Konzultácie
 4 stretnutia počas semestra: napr. 1. a 3. týždeň, potom pre kontrolu 6.týždeň a pred odovzdaním 12.týždeň. Len podľa potreby aj medzitým.


## Agenda

   - formálnych požiadaviek na výstup z DP1 (rozsah, potrebnú podmnožinu častí oproti celej diplomovke, plán práce)
   - tvojich odporúčaní k členeniu obsahu - podľa osnovy vychadzajúc z výskumného zámeru.
   - doplňujúce k otázky ohľadom "strategického" smerovania práce - ako to napr. vyzerá s tou avizovanou spoluprácou s továrňou, testovací rig, potenciálna integrácia s loghubom atď. Je to zrejme ešte ďaleko, kým sa prakticky k tomu dopracujem, ale stále mi nie je presne jasná predstava prostredia v ktorom by zariadenie pôsobilo - určitý prípad použitia (nepotrebujem návod, len schematicky).
   - aká existuje miera voľnosti v mojom prístupe k problematike -  či v analýze pokryjem aj bližšie 1 konkrétny stroj alebo má ísť o všeobecný teoretický pohľad na algoritmy



## Osnova

   - **Physics of rotating machines and fault types**
   		- fyzikálny popis ako pohyblivé diely spôsobujú vibrácie 
   		- (časti stroja na opis ako sú hriadeľ, ozubenie, remeň, ložiská vyberiem podľa konzultácie), 
   		- typy poškodení a ako sa prejavujú - pokúsim sa tomu dať viac dátový než strojnícky pohľad. - harmonická charakteristika
   - **Condition monitoring and predictive maintenance nowadays** 
   		- ISO štandardy - metriky (magnitúda), dôsledky ako upevnenie sezorov ovplyvný charakteristiku - môžeme si myslieť iné ako štandardy
   		- proces a postupy vyhodnocovania pochôdzových meraní vrátane softvéru,
   		- rozmiestnenie senzorov
   - **IoT in industry** - aká je perspektíva uplatnenia IoT pri vibrodiagnostike
   		- čo sa už skúšalo, prakticky nasadilo,  - hw vynechať
   		- možno protokoly a formáty dát 
   - **Feature extraction** - spektrálne char.
   		- čo sa bude ukladať
   		- čo posielať priebežne cez sieť,
   		- hlavne ako tie črty získame:
   			- time-frequency analysis - PSD
   			- harmonics and sideband,
   			- filtering:
   			- BSS a **independent component analysis (ICA)** - metriky kurtosis, negentropy, MI
   			- známe prístupy ako MFCC
   - **Weak supervision and Active learning** 
   - Určiť si čo sú clustre vôbec
   - na diagnostiku poškodení podľa pár označených vzoriek a ich zovšeobecnenie na štatistický model - napr. cez algoritmy zhlukovania (alebo iné čo nájdem v literatúre) - **STREAMING ALGORITMS** (Sketching algorithms - Cluster-reduce) - DBSCAN (+k-d tree), BIRCH
   

## Postup - nástroje

1. Meranie 1 senzorom akcelerácie
2. Výpočet BSS (FastICA) zo vzoriek - alebo iný spôsob dátového oddelenia mechanických častí stroja cez metriky (Kurtosis / MI / Negentropy) - vieme iba stacionárne a šum (čiže aj forma filteringu)
3. (alebo vymeniť s krokom 2) Vytvorenie sumarizačných čŕt (pre časť stroja) z frekvenčného spektra - WT + SQ - segmentácia TF spektra (viď. MFCC, harmonics, peaks) - signatúru

3. Synchronizácia senzorov na jednom stroji (alebo do master jednotky) - iba ako výpočtový krok distribuovaného výpočtu vo všeobecnom algoritme


5. Charakterizovanie vývoja stavu častí cez reprezentáciu v clusteroch - real-time
   Sketch - BIRCH, DBSCAN, Cluster-Reduce, 
6. Query stavu stroja v podobe - tento frekv. vzor je označený za hriadeľ, ten má takýto vývoj amplitúd -  harmonické a fáza naznačujú že by mohlo ísť o nevyváženosť
7. Feedback - dať k dispozícii okraje clusterov a informácie tak aby ich operátor vedel priradiť k jednej alebo druhej kategórii

Definovať si časti stroja
Definovať si typy (kategórie porúch)
Definovať si ukazovateľe pre poruchu

"statistical methods to extract signal features that are highly correlated with the wear status of the band saw blade"

## Postup - problém

1. Napr. je hriadeľ pevná vs. hriadeľ s vôlou (binárna klasifikácia - spojitý prechod - ale threashold stanovuje kedy výmena)
2. Predpoklad: existuje iba táto chyba
3. V čase meriam frekvenčné spektrá - odhad PSD (Welch+Exponential smoothing) - líšia sa bez/s poškodenia.
4. Na ich odlíšenie podobnosť spektier (výpočet a posielanie tento miery bude predmetom optimalizácie - adaptívne vzorkovanie) 
	- https://makeabilitylab.github.io/physcomp/signals/ComparingSignals/index.html napr. cross-korelácia (https://en.wikipedia.org/wiki/Cross-correlation), energia (je nelokalizovaná), Spectral flatness
	- https://www.researchgate.net/post/How_to_measure_waveform_similarity_between_two_waves
	- Spektrálna obálka + Quasiperiodicity zachytená v harmonických
	- Článok: ROBUST SIMILARITY METRICS BETWEEN AUDIO SIGNALS BASED ON ASYMMETRICAL SPECTRAL ENVELOPE MATCHING
	- Článok: Research on online intelligent monitoring system of band saw blade wear status based on multi‑feature fusion of acoustic emission signals
	- Článok: An Adaptive Spectrum Segmentation Method to Optimize Empirical Wavelet Transform for Rolling Bearings Fault Diagnosis
5. Pri jednej chybe stačí aj iná jednoduchšia metrika
6. Ak detegujeme množinu poškodení pre 1 stroj (stavy môžu byť aj kombinácie čiže 2^(počet typov poškodení)) = {OK} + {Vôla, Nevyváženosť, Mimoosovosť}, tak dostaneme iné klastre podľa metriky podobnosti frekvenčných spektier. Expert oanotuje ktorý klaster je aké poškodenie alebo kominácia poškodení. - distribúcia značiek

## Výskumné otázky


1. Which time-frequency features can be extracted from vibrational signals to pro-
vide an accurate record of machinery faults?
	- *Popis signatúry udalosti pre jednu mechanickú časť stroja získanú po BSS*
2. What are the savings in transmission bandwidth when chosen signal features are used in comparison to raw sampled measurement or lossless compression
techniques?
	- *Porovnaj úspory keď predspracovanie forma bude až posielaná*
3. How can the machinery faults be continuously identified based on collected
events?
	- *tá spätná väzba nad clusetrovaním*
	
- 20 - 30 strán - analýza a základ návhu	
- Výroba šalovacích dosiek doka.
- Exkurzia do fabriky - doka - 16.3. / 24.3. piatok - bystrica
- Iný pohľad - inžiniersky - nasadiť výsledky článku vo fabrike.
- tooth per inch, určiť si max. meracie frekvencie (typ píly po návšteve, zobrať mic, akcelerometre na exkurziu)


- Určiť si čo vôbec budú klastre z akýc čŕt
- PCA, Elipsové klastrovanie - separačné a klastrovacie algoritmy
- Dlhodobá amplitúda - trend
	+ ako dodatok tie konkrétne poruchy (odteraz 10 sekúnd merania po evente alebo snapshot)


## Python knižnice
- numpy
- scipy
	+ https://docs.scipy.org/doc/scipy/tutorial/fft.html
	+ https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
	+ https://gist.github.com/fasiha/957035272009eb1c9eb370936a6af2eb
- pandas
- jupyterlab
- matplotlib
- scikit-learn
	+ https://scikit-learn.org/stable/modules/decomposition.html
	+ https://scikit-learn.org/stable/modules/clustering.html
	+ https://scikit-learn.org/stable/modules/mixture.html
	+ https://scikit-learn.org/stable/modules/semi_supervised.html
	+ Clustering metrics: Jaccard Index podobnosti: 
		- https://www.statology.org/jaccard-similarity/
		- https://en.wikipedia.org/wiki/Jaccard_index
		- https://lilianweng.github.io/posts/2021-12-05-semi-supervised/
		- https://blog.roboflow.com/what-is-semi-supervised-learning/ (Label propagation, MixMatch)
		- https://festinais.medium.com/mixmatch-a-holistic-approach-to-semi-supervised-learning-1480b56f96b7
		- https://en.wikipedia.org/wiki/Fuzzy_clustering
- PyWavelets
- detecta
	+ https://github.com/demotu/detecta
- ssqueezepy
	+ https://dsp.stackexchange.com/questions/71398/synchrosqueezing-wavelet-transform-explanation/71399#71399
- fcwt
	+ https://github.com/fastlib/fCWT
- spafe
	+ https://spafe.readthedocs.io/en/latest/frequencies/fundamental_frequencies.html

## Datasets
- https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
- https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft
- https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets

## Stroje
- Band saw, Circular saw in Sawmill
- Water centrifugal pump in Water pumping station
- Washing machine in Laundry place
- Air compressor
