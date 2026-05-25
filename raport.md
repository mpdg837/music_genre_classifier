# Raport finalny - interpretowalność klasyfikatora gatunku na muzyce symbolicznej

Michał Podgajny 311412  
Miłosz Andryczuk  
Aleksander Szymczyk 325239

## Opis projektu

Projekt dotyczy klasyfikacji gatunku muzycznego na podstawie plików MIDI oraz analizy tego, jakie cechy muzyczne model uznaje za charakterystyczne dla poszczególnych gatunków. Głównym celem nie było wyłącznie uzyskanie jak najwyższej skuteczności klasyfikacji, ale także przejście od samej predykcji do interpretacji decyzji modelu.

Inspiracją była praca Foscarina et al. (2022), w której metoda TCAV została wykorzystana do interpretacji modeli rozpoznających kompozytorów. W tym projekcie analogiczny pomysł przeniesiono na klasyfikację gatunków muzycznych. Zamiast analizować pojedyncze nuty lub aktywacje bez znaczenia muzycznego, zdefiniowano zrozumiałe koncepty, takie jak wysoka gęstość nut, niski rejestr, długa średnia długość nut czy nieregularny rytm.

W ramach projektu przygotowano pełny pipeline eksperymentalny:

- pobranie i preprocessing danych MIDI,
- ekstrakcję cech symbolicznych,
- analizę eksploracyjną zbioru,
- trening klasycznych modeli uczenia maszynowego,
- trening modeli neuronowych opartych na piano-rollu i sekwencjach nut,
- fine-tuning modelu MusicBERT,
- analizę interpretowalności metodą TCAV.

## Funkcjonalność programu

System ma charakter badawczo-eksperymentalny. Jest uruchamiany z poziomu linii poleceń i konfigurowany za pomocą plików Hydra, co pozwala odtwarzać eksperymenty oraz porównywać różne warianty modeli.

Najważniejsze funkcjonalności obejmują:

- wczytywanie i walidację plików MIDI,
- konwersję plików MIDI do wspólnej reprezentacji numerycznej,
- budowę zbiorów danych dla modeli klasycznych, MuSeReNet, Transformera i MusicBERT,
- ekstrakcję cech muzycznych wykorzystywanych zarówno do klasyfikacji, jak i do definicji konceptów TCAV,
- trening i ewaluację modeli klasyfikujących gatunek,
- rejestrowanie wyników w Weights & Biases,
- obsługę konfiguracji eksperymentów przez Hydra,
- uruchamianie eksperymentów na klastrze przez skrypty Slurm,
- analizę interpretowalności modelu metodą TCAV,
- generowanie tabel, wykresów i podsumowań wyników.

W przeprowadzonych eksperymentach wykorzystano jeden główny dataset, XMIDI. Architektura kodu pozwala natomiast na zastosowanie tego samego pipeline'u do kolejnych zbiorów MIDI, co byłoby naturalnym rozszerzeniem projektu w kierunku porównywania "definicji gatunku" zakodowanych w różnych datasetach.

## Użyte narzędzia

Do przetwarzania plików MIDI wykorzystano bibliotekę `partitura`, ponieważ dobrze wspiera analizę muzyki symbolicznej i była zgodna z kierunkiem pracy Foscarina et al. W projekcie wykorzystano także:

- `pandas` i `numpy` - przetwarzanie danych tabelarycznych oraz numerycznych,
- `partitura` - wczytywanie MIDI i ekstrakcja reprezentacji symbolicznych,
- `scikit-learn` - modele klasyczne, metryki i klasyfikator CAV,
- `PyTorch` - implementacja i trening modeli neuronowych,
- `transformers` - integracja modelu MusicBERT przez Hugging Face,
- `miditok` - tokenizacja MIDI do reprezentacji REMI dla MusicBERT,
- `Captum` - implementacja TCAV,
- `matplotlib` - wizualizacja wyników,
- `Hydra` - zarządzanie konfiguracjami eksperymentów,
- `Weights & Biases` - logowanie metryk i porównywanie runów,
- `Slurm` - uruchamianie dłuższych eksperymentów na GPU,
- `ruff`, `pytest`, `uv` i `make` - organizacja środowiska, formatowanie, testy i automatyzacja zadań.

## Dane

Do eksperymentów wykorzystano zbiór **XMIDI**. Pliki MIDI mają etykiety gatunku oraz emocji, dzięki czemu można analizować zarówno cechy stylistyczne, jak i rozkład dodatkowych metadanych. W tym projekcie głównym zadaniem była klasyfikacja gatunku, dlatego etykiety emocji wykorzystano wyłącznie w analizie eksploracyjnej.

Aktualnie przetworzony zbiór użyty w eksperymentach zawiera **52 421 próbek**. Dane nie zawierają braków w kolumnach metadanych i nie zawierają zduplikowanych identyfikatorów `sample_id`.

### Rozkład gatunków

Zbiór jest niezbalansowany. Najliczniejsze klasy to `rock`, `pop` i `country`, natomiast najmniej liczna jest klasa `traditional`.

| Gatunek | Liczba próbek | Udział [%] |
|---|---:|---:|
| rock | 13 007 | 24.81 |
| pop | 12 380 | 23.62 |
| country | 11 539 | 22.01 |
| jazz | 7 697 | 14.68 |
| classical | 6 047 | 11.54 |
| traditional | 1 751 | 3.34 |

Nierównowaga klas jest jednym z głównych ograniczeń eksperymentu. Szczególnie problematyczna jest klasa `traditional`, która ma ponad siedmiokrotnie mniej próbek niż klasy `rock` lub `pop`.

### Rozkład emocji

Etykiety emocji nie były celem klasyfikacji, ale pokazują dodatkową strukturę datasetu.

| Emocja | Liczba próbek | Udział [%] |
|---|---:|---:|
| exciting | 10 227 | 19.51 |
| warm | 7 284 | 13.90 |
| happy | 6 460 | 12.32 |
| romantic | 6 182 | 11.79 |
| funny | 6 076 | 11.59 |
| sad | 4 396 | 8.39 |
| angry | 4 244 | 8.10 |
| lazy | 2 242 | 4.28 |
| quiet | 2 117 | 4.04 |
| fear | 1 772 | 3.38 |
| magnificent | 1 421 | 2.71 |

### Preprocessing

Każdy plik MIDI został wczytany i zapisany do formatu `.npz` zawierającego podstawowe informacje o nutach:

- `pitch` - wysokość dźwięku,
- `onset_sec` - czas rozpoczęcia nuty,
- `duration_sec` - czas trwania nuty,
- `velocity` - dynamika,
- `track` - numer ścieżki,
- `channel` - kanał MIDI.

Na podstawie tych danych utworzono kilka reprezentacji:

- **cechy tabularne** dla modeli klasycznych,
- **piano-roll** dla MuSeReNet,
- **macierz nut** dla MIDI Transformera,
- **tokeny REMI** dla MusicBERT.

### Cechy muzyczne

Dla modeli klasycznych oraz do definicji konceptów TCAV wyekstrahowano cechy opisujące strukturę muzyczną utworów:

- liczba nut,
- czas trwania utworu,
- gęstość nut,
- średnia i maksymalna polifonia,
- średnia, odchylenie i zakres wysokości dźwięków,
- statystyki długości nut,
- statystyki velocity,
- średnia i odchylenie inter-onset interval,
- histogram klas wysokości dźwięków.

Analiza eksploracyjna pokazała, że cechy te niosą informację o gatunku, ale same nie rozdzielają klas idealnie. Na projekcji PCA klasy częściowo się nakładały, co potwierdza, że zadanie klasyfikacji jest nietrywialne.

Najważniejsze obserwacje z eksploracji danych:

- `pop` wyróżniał się wysoką liczbą nut i dużą gęstością zdarzeń,
- `country` miał relatywnie wysoką średnią polifonię,
- `classical` częściej wykazywał wyższy rejestr i dłuższe wartości rytmiczne,
- `traditional` miał niższą gęstość nut i spokojniejszy przebieg,
- `jazz` wykazywał większą zmienność wysokości dźwięków,
- profile pitch-class różniły się między gatunkami, choć nie wystarczały do jednoznacznego rozróżnienia klas.

Wygenerowane wykresy EDA znajdują się w katalogu `reports/figures/xmidi_eda`.

## Reprezentacje wejściowe

### Cechy tabularne

Modele klasyczne otrzymują jeden wektor cech dla całego utworu. Jest to reprezentacja mało kosztowna obliczeniowo i łatwa do interpretacji, ale traci szczegółową informację o kolejności zdarzeń muzycznych.

### Piano-roll

Reprezentacja piano-roll opisuje aktywność wysokości dźwięków w czasie. Oś pozioma odpowiada czasowi, a oś pionowa wysokościom MIDI. Taka reprezentacja zachowuje lokalne wzorce melodyczno-rytmiczne i może być traktowana podobnie do obrazu, dlatego dobrze pasuje do modeli konwolucyjnych.

W projekcie piano-roll wykorzystano jako wejście modelu MuSeReNet.

### Macierz nut

Dla MIDI Transformera każdy utwór reprezentowany jest jako sekwencja nut. Każda nuta opisana jest sześcioma cechami:

| Cecha | Znaczenie |
|---|---|
| `pitch` | wysokość dźwięku MIDI |
| `onset_sec` | czas rozpoczęcia nuty |
| `duration_sec` | czas trwania nuty |
| `velocity` | dynamika |
| `track` | numer ścieżki |
| `channel` | kanał MIDI |

Sekwencje mają różną długość, dlatego są przycinane lub uzupełniane paddingiem do stałej długości `max_notes`. Model otrzymuje także maskę wskazującą rzeczywiste nuty.

### REMI i MusicBERT

MusicBERT wymaga tokenizacji MIDI do reprezentacji symbolicznej. W projekcie użyto tokenizacji REMI, która zamienia muzykę na sekwencję tokenów opisujących między innymi pozycje rytmiczne, wysokości, długości i velocity. Długie utwory są dzielone na okna tokenów. W treningu stosowane jest przycinanie do maksymalnej długości, a w ewaluacji deterministyczne okna, dzięki czemu model może oceniać dłuższe pliki w bardziej stabilny sposób.

## Modele

### Modele klasyczne

Jako punkt odniesienia wykorzystano modele trenowane na cechach tabularnych:

- Logistic Regression,
- Linear SVC,
- SVC,
- KNN,
- Random Forest,
- MLP.

Ich celem było sprawdzenie, ile informacji o gatunku da się odzyskać z prostych, ręcznie zaprojektowanych cech symbolicznych.

### MuSeReNet

MuSeReNet jest modelem konwolucyjnym działającym na piano-rollu. Model analizuje lokalne wzorce w przebiegu utworu, takie jak powtarzalne motywy, zagęszczenia nut, zmiany rejestru i fragmenty o większej polifonii. W projekcie przetestowano wariant bazowy oraz większy wariant modelu.

### MIDI Transformer

MIDI Transformer analizuje sekwencję nut opisaną cechami numerycznymi. W założeniu powinien modelować relacje czasowe między zdarzeniami muzycznymi. W praktyce długie sekwencje MIDI i duża liczba nut w utworach okazały się trudne dla modelu trenowanego od zera.

### MusicBERT

MusicBERT jest modelem wstępnie trenowanym na muzyce symbolicznej. W projekcie użyto go przez Hugging Face, dodając własną głowicę klasyfikacyjną do predykcji gatunku. Przetestowano dwa warianty:

- **MusicBERT full fine-tuning** - aktualizowane są wagi encodera i głowicy,
- **MusicBERT frozen head** - encoder jest zamrożony, trenowana jest głównie głowica klasyfikacyjna.

Pełny fine-tuning MusicBERT dał najlepszy wynik, ale był zdecydowanie najbardziej kosztowny obliczeniowo.

## Ewaluacja modeli

Modele oceniano na tym samym zbiorze walidacyjnym zawierającym **10 485 przykładów**. Ze względu na silne niezbalansowanie klas główną metryką porównawczą jest **macro F1**, ponieważ traktuje wszystkie klasy równorzędnie i nie jest zdominowana przez najliczniejsze gatunki.

Dla modeli neuronowych raportowane są najlepsze wartości `val_f1_macro` zaobserwowane w trakcie treningu, a nie wyłącznie wynik z ostatniej epoki.

| Model / run | Reprezentacja | Best val macro-F1 | Epoka | Val accuracy |
|---|---|---:|---:|---:|
| MusicBERT full fine-tuning | REMI tokens | **0.588** | 5 | 0.622 |
| Random Forest | cechy tabularne | **0.562** | - | 0.614 |
| MuSeReNet, większy wariant | piano-roll | **0.507** | 36 | 0.551 |
| MuSeReNet, wariant bazowy | piano-roll | **0.485** | 46 | 0.531 |
| MusicBERT frozen head | REMI tokens | **0.456** | 4 | 0.507 |
| SVC | cechy tabularne | **0.450** | - | 0.471 |
| MLP | cechy tabularne | **0.436** | - | 0.485 |
| KNN | cechy tabularne | **0.431** | - | 0.472 |
| Logistic Regression | cechy tabularne | **0.395** | - | 0.412 |
| Linear SVC | cechy tabularne | **0.389** | - | 0.440 |
| MIDI Transformer | sekwencja nut | **0.387** | 20 | 0.465 |

Najlepszy wynik uzyskał MusicBERT po pełnym fine-tuningu. Potwierdza to wartość wykorzystania wstępnie trenowanych encoderów dla muzyki symbolicznej. Jednocześnie bardzo dobry wynik Random Forest pokazuje, że ręcznie zaprojektowane cechy symboliczne zawierają dużo informacji o gatunku. Jest to ważne także z punktu widzenia interpretowalności, ponieważ te same cechy można wykorzystać do definiowania muzycznych konceptów.

### Modele klasyczne

| Model | Train accuracy | Val accuracy | Train macro-F1 | Val macro-F1 | Różnica F1 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.996 | 0.614 | 0.996 | 0.562 | 0.434 |
| SVC | 0.511 | 0.471 | 0.497 | 0.450 | 0.047 |
| MLP | 0.588 | 0.485 | 0.537 | 0.436 | 0.101 |
| KNN | 0.605 | 0.472 | 0.557 | 0.431 | 0.125 |
| Logistic Regression | 0.420 | 0.412 | 0.400 | 0.395 | 0.005 |
| Linear SVC | 0.442 | 0.440 | 0.390 | 0.389 | 0.001 |

Random Forest uzyskał najlepszy wynik wśród modeli klasycznych, ale bardzo duża różnica między wynikiem treningowym i walidacyjnym wskazuje na overfitting. SVC osiągnął niższy wynik, ale generalizował stabilniej. Modele liniowe miały najmniejszą różnicę między treningiem i walidacją, ale były zbyt proste, aby uchwycić bardziej złożone zależności.

### MuSeReNet i Transformer

| Metryka | MuSeReNet bazowy | MIDI Transformer |
|---|---:|---:|
| Accuracy | 0.53 | 0.46 |
| Macro precision | 0.50 | 0.40 |
| Macro recall | 0.48 | 0.39 |
| Macro F1 | 0.48 | 0.39 |
| Weighted F1 | 0.52 | 0.45 |

MuSeReNet był wyraźnie skuteczniejszy niż testowany MIDI Transformer. Sugeruje to, że w tej konfiguracji reprezentacja piano-roll była łatwiejsza do wykorzystania niż długa sekwencja nut. Transformer miał szczególnie duży problem z klasami `jazz` i `traditional`.

| Gatunek | Support | MuSeReNet F1 | Transformer F1 | Różnica |
|---|---:|---:|---:|---:|
| classical | 1 209 | 0.62 | 0.63 | -0.01 |
| country | 2 308 | 0.54 | 0.49 | +0.05 |
| jazz | 1 540 | 0.52 | 0.26 | +0.26 |
| pop | 2 476 | 0.52 | 0.48 | +0.04 |
| rock | 2 602 | 0.53 | 0.47 | +0.06 |
| traditional | 350 | 0.15 | 0.00 | +0.15 |

Najtrudniejszą klasą była `traditional`. Ma ona najmniejszy support i oba modele miały problem z jej rozpoznaniem. Niska wartość macro F1 względem weighted F1 potwierdza, że modele radzą sobie lepiej z klasami liczniejszymi niż z klasą mniejszościową.

### MusicBERT

MusicBERT uzyskał najlepszy wynik całego projektu:

- best val macro-F1: **0.588**,
- val accuracy: **0.622**,
- najlepsza zaobserwowana epoka: **5**.

Wynik ten był lepszy od Random Forest o około 0.027 macro-F1 oraz od bazowego MuSeReNet o około 0.103 macro-F1. Poprawa nie była jednak darmowa: pełny fine-tuning MusicBERT był najbardziej kosztownym eksperymentem i wymagał długiego czasu treningu na GPU. Wariant z zamrożonym encoderem był tańszy i bardziej stabilny organizacyjnie, ale osiągnął niższy wynik macro-F1 równy 0.456.

## Analiza błędów

Najważniejszym źródłem błędów była nierównowaga klas. Klasa `traditional` stanowi tylko 3.34% zbioru, co przekłada się na niski support w walidacji i bardzo słabe wyniki dla tej klasy. Modele częściej poprawnie rozpoznawały klasy liczne, takie jak `rock`, `pop` i `country`.

Drugim problemem jest częściowe nakładanie się cech muzycznych między gatunkami. Analiza PCA oraz wyniki modeli klasycznych pokazują, że cechy symboliczne są informatywne, ale nie tworzą prostych, liniowo rozdzielnych grup. Dotyczy to szczególnie gatunków popularnych, które mogą dzielić podobne schematy rytmiczne, rejestry i gęstości nut.

W praktyce oznacza to, że błędne klasyfikacje nie wynikają wyłącznie z niedoskonałości architektury, ale także z natury danych: gatunki muzyczne nie są kategoriami ostro oddzielonymi, a dataset może kodować uproszczone lub specyficzne dla źródła definicje gatunków.

## Interpretowalność TCAV

Do interpretacji modelu wykorzystano metodę **TCAV** (*Testing with Concept Activation Vectors*). TCAV sprawdza, czy przesunięcie reprezentacji modelu w kierunku określonego konceptu zwiększa wynik dla danej klasy. Dzięki temu można pytać model o pojęcia zrozumiałe muzycznie, np. czy wysoka gęstość nut wspiera klasyfikację jako `pop`.

### Koncepty

Koncepty zdefiniowano na podstawie cech symbolicznych:

- `high_note_density` i `low_note_density`,
- `high_polyphony`,
- `wide_pitch_range`,
- `high_pitch_register` i `low_pitch_register`,
- `strong_velocity`,
- `long_notes` i `short_notes`,
- `irregular_ioi` i `regular_ioi`.

Dla każdego konceptu utworzono manifest próbek reprezentujących dany ogon rozkładu cechy. Następnie porównywano je z losowymi kontrolami. CAV był trenowany jako liniowy klasyfikator rozdzielający koncept od kontroli.

### Konfiguracja TCAV

Ukończoną analizę TCAV przeprowadzono dla modelu MuSeReNet. Ewaluowano dwie warstwy klasyfikatora:

- `classifier.1`,
- `classifier.2`.

Łącznie wykonano **132 testy**:

- 11 konceptów,
- 6 klas,
- 2 warstwy.

Każdy koncept porównano z 10 losowymi kontrolami. Do oceny istotności wykorzystano test statystyczny względem wartości 0.5 oraz skorygowany próg istotności `alpha = 0.000379`.

### Wyniki TCAV

Spośród 132 testów **20 było istotnych statystycznie**:

- 10 pozytywnych,
- 10 negatywnych,
- wszystkie istotne wyniki pojawiły się w warstwie `classifier.1`,
- dla `classifier.2` nie uzyskano istotnych wyników po korekcji.

| Gatunek | Koncept | Warstwa | Mean sign count | Kierunek |
|---|---|---|---:|---|
| classical | high_pitch_register | classifier.1 | 0.938 | pozytywny |
| classical | irregular_ioi | classifier.1 | 0.953 | pozytywny |
| classical | long_notes | classifier.1 | 0.952 | pozytywny |
| classical | low_note_density | classifier.1 | 0.898 | pozytywny |
| country | high_note_density | classifier.1 | 0.271 | negatywny |
| country | low_pitch_register | classifier.1 | 0.687 | pozytywny |
| country | regular_ioi | classifier.1 | 0.288 | negatywny |
| jazz | high_note_density | classifier.1 | 0.191 | negatywny |
| jazz | high_pitch_register | classifier.1 | 0.189 | negatywny |
| jazz | low_note_density | classifier.1 | 0.150 | negatywny |
| jazz | short_notes | classifier.1 | 0.174 | negatywny |
| pop | high_note_density | classifier.1 | 0.839 | pozytywny |
| pop | high_pitch_register | classifier.1 | 0.221 | negatywny |
| pop | long_notes | classifier.1 | 0.221 | negatywny |
| rock | high_polyphony | classifier.1 | 0.261 | negatywny |
| rock | long_notes | classifier.1 | 0.214 | negatywny |
| rock | short_notes | classifier.1 | 0.744 | pozytywny |
| rock | strong_velocity | classifier.1 | 0.692 | pozytywny |
| traditional | high_pitch_register | classifier.1 | 0.791 | pozytywny |
| traditional | long_notes | classifier.1 | 0.721 | pozytywny |

Wyniki są zgodne z częścią obserwacji z analizy eksploracyjnej. Dla `pop` istotnie pozytywny był koncept wysokiej gęstości nut. Dla `rock` pozytywny wpływ miały krótsze nuty i silniejsze velocity. Dla `classical` model był wrażliwy na wysoki rejestr, długie nuty, niższą gęstość oraz nieregularność odstępów czasowych. Dla `traditional` istotne były długie nuty i wyższy rejestr.

Należy jednak podkreślić, że TCAV opisuje **wrażliwość konkretnego modelu**, a nie obiektywną definicję gatunku muzycznego. Wyniki mówią więc, jakie koncepty MuSeReNet wykorzystuje w swoich reprezentacjach, a nie które cechy są uniwersalnie charakterystyczne dla gatunków.

### Wizualizacje TCAV

Poniżej umieszczono najważniejsze wykresy wygenerowane dla analizy TCAV. Heatmapy pokazują średnie wyniki TCAV dla konceptów, klas i warstw, natomiast wykresy słupkowe pozwalają prześledzić najważniejsze koncepty dla konkretnych gatunków.

**Rysunek 1. Zbiorcza heatmapa TCAV dla analizowanych warstw**

![Zbiorcza heatmapa TCAV](reports/tcav/tcav_heatmap_all_layers_mean.png)

**Rysunek 2. Heatmapa TCAV dla warstwy `classifier.1`**

![Heatmapa TCAV dla classifier.1](reports/tcav/tcav_heatmap_classifier_1.png)

Warstwa `classifier.1` była najważniejsza interpretacyjnie, ponieważ wszystkie istotne statystycznie wyniki po korekcji pojawiły się właśnie w tej warstwie. Dlatego poniżej pokazano przykłady najciekawszych klas.

**Rysunek 3. Najważniejsze koncepty dla klasy `classical`**

![Najważniejsze koncepty TCAV dla classical](reports/tcav/tcav_top_concepts_classifier_1_classical.png)

Dla klasy `classical` model był szczególnie wrażliwy na długie nuty, wysoki rejestr, niższą gęstość nut oraz nieregularne odstępy czasowe.

**Rysunek 4. Najważniejsze koncepty dla klasy `pop`**

![Najważniejsze koncepty TCAV dla pop](reports/tcav/tcav_top_concepts_classifier_1_pop.png)

Dla klasy `pop` najwyraźniejszym pozytywnym konceptem była wysoka gęstość nut, natomiast długie nuty i wysoki rejestr działały w kierunku przeciwnym.

**Rysunek 5. Najważniejsze koncepty dla klasy `rock`**

![Najważniejsze koncepty TCAV dla rock](reports/tcav/tcav_top_concepts_classifier_1_rock.png)

Dla klasy `rock` pozytywnie działały krótsze nuty i silniejsze velocity, a negatywnie długie nuty oraz wysoka polifonia.

**Rysunek 6. Najważniejsze koncepty dla klasy `traditional`**

![Najważniejsze koncepty TCAV dla traditional](reports/tcav/tcav_top_concepts_classifier_1_traditional.png)

Klasa `traditional` była najtrudniejsza klasyfikacyjnie, dlatego jej wykres jest szczególnie istotny diagnostycznie. Model wiązał ją głównie z długimi nutami i wyższym rejestrem, mimo że liczba przykładów tej klasy była mała.

## Zużyte zasoby

Eksperymenty uruchamiano na klastrze z użyciem GPU NVIDIA A100. Najbardziej kosztowne były eksperymenty z MusicBERT oraz TCAV, ponieważ wymagają wielokrotnego przetwarzania długich sekwencji MIDI i obliczania aktywacji dla wielu konceptów, klas i warstw.

Łączny koszt obliczeniowy oszacowano na około **75 GPU-godzin**. Wartość ta obejmuje zarówno udane treningi, jak i eksperymenty przerwane przez limity czasu lub wykorzystane do strojenia konfiguracji.

## Ograniczenia

Najważniejsze ograniczenia projektu:

- główne eksperymenty wykonano na jednym zbiorze, XMIDI,
- klasa `traditional` jest silnie niedoreprezentowana,
- pełny fine-tuning MusicBERT jest kosztowny czasowo,
- modele trenowane od zera osiągnęły niższe wyniki niż MusicBERT i Random Forest,
- TCAV został ukończony i zaraportowany dla MuSeReNet; analogiczna analiza dla MusicBERT wymaga dodatkowego czasu obliczeniowego,
- analiza błędów została wykonana głównie na poziomie klas i metryk, a nie jako ręczna analiza pojedynczych utworów.

## Wnioski

Projekt pokazał, że klasyfikacja gatunku na danych MIDI jest możliwa, ale trudna ze względu na nierównowagę klas i częściowe nakładanie się cech muzycznych między gatunkami.

Najlepszy wynik klasyfikacyjny uzyskał **MusicBERT full fine-tuning** z macro-F1 równym **0.588**. Potwierdza to, że wstępnie trenowane modele muzyczne są bardziej efektywne niż architektury trenowane od zera na tym zbiorze.

Bardzo mocnym punktem odniesienia okazał się **Random Forest**, który na ręcznie zaprojektowanych cechach osiągnął macro-F1 równe **0.562**. Oznacza to, że cechy symboliczne, takie jak gęstość nut, polifonia, rejestr i długość nut, zawierają dużą część informacji potrzebnej do rozpoznawania gatunku.

Spośród modeli trenowanych od zera lepszy był **MuSeReNet** oparty na piano-rollu. MIDI Transformer miał większe problemy z długimi sekwencjami i klasami mniejszościowymi.

Analiza TCAV umożliwiła przejście od samej skuteczności modelu do interpretacji jego decyzji. Wyniki wskazały między innymi, że MuSeReNet wiązał `pop` z wysoką gęstością nut, `rock` z silniejszą dynamiką i krótszymi nutami, a `classical` z dłuższymi nutami, wyższym rejestrem i niższą gęstością.

Najważniejszy rezultat projektu jest więc dwojaki: zbudowano pipeline do klasyfikacji gatunków MIDI oraz pokazano, że decyzje modelu można analizować przez muzycznie zrozumiałe koncepty, a nie tylko przez abstrakcyjne aktywacje sieci.

## Dalsze prace

Naturalne rozszerzenia projektu obejmują:

- dokończenie i porównanie TCAV dla MusicBERT,
- dodanie kolejnych datasetów MIDI i porównanie zakodowanych w nich definicji gatunków,
- dokładniejszą analizę pojedynczych próbek błędnie sklasyfikowanych,
- augmentację lub oversampling klasy `traditional`,
- testowanie bogatszych konceptów muzycznych, np. związanych z harmonią, metrum, repetytywnością i konturem melodii,
- porównanie wyników TCAV między modelem symbolicznym, konwolucyjnym i pretrained encoderem.
