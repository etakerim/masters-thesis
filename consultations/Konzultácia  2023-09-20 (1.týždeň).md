
1. Krátke zhodnotenie DP1
2. Ciele: Vytýčenie požadovaných (minimálnych a optimálnych) cieľov v predmete DP2 (ako podmnožina z časti návrhu riešenia z reportu k DP1)
3. Spolupráce: Aktuálna situácia okolo HW akcelerometra a zberu údajov z fabriky
4. Metodika: Aké metriky a metódy aplikovať na dataset (Na začiatok by som chcel začať s implementáciou supervised ML modelu nad MaFaulDa datasetom podľa časti 3.1 z reportu. Potreboval by som poradiť s výberom metrík na spracovanie a pomoc s vytýčením miľníkov, že čo skúmať, aby som neodbiehal od pointy alebo netočil sa v kruhoch).

+ Dátovú pipeline v Pythone na Datasete
	+ Spočítať kompresný pomer (PCM a DPCM voči štatistaikám)
	+ 1 os na jednom mieste, MauFaulDa, rozdeliť z poruchy na trénovaciu a testovaciu a nasypať
	+ low pass na 10kHz a potom downsampling (aspoň 2x voči tomu čo chcem pozerať)
	+ 2. krok ložiská
	+ Rýchlosti rôzne?
	+ transformácia n-rozmerná
	+ Viacrozmerné (do budúcna)?
	+ Anomália - normálalna vs porucha (relatívne ) - podľa modelu
		+ Ak nebudú dostačne separovateľné, tak zbehnúť PCA
	+ EWT nakoniec ako čerešničku
	+ Pre fourierku si zvoliť dostatočne odlišné parametre -> výlsedok modelu (obhájiť prečo som niečo nastaviť nejako empiricky) 
		+ Hill climbing samoučiaci algoritmus na nájdenie parametrou -> odmena pre generatívny algoritmus
		+ Dostať sa na rozumne málo stúpajúcu krivku
+ Hardvér (začať implementáciu)
+ Implementácia (čo som za semester spravil), Ako budeme testovať/overovať?, Plán práce,  Záver
+ --- Dohodnúť či sa dá niečo merať v DC Digitalise.
+ 11.10. o 9:00