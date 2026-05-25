# Raport finalny - Interpretowalność klasyfikatora gatunku na muzyce symbolicznej.

Michał Podgajny 311412

Miłosz Andryczuk 

Aleksander Szymczyk

## Opis projektu

Projekt polega na wytrenowaniu klasyfikatora gatunku na wybranych datasetach MIDI i zastosowaniu metod concept-based interpretability do analizy, które cechy model uznaje za charakterystyczne dla każdego gatunku. Inspiracją jest praca Foscarina et al. (2022), gdzie TCAV zastosowano do klasyfikacji kompozytorów. Projekt przenosi to podejście na gatunki, definiując odpowiednie koncepty muzyczne. Dodatkowym elementem jest porównanie "definicji gatunku" zakodowanych w różnych datasetach oraz analiza próbek błędnie sklasyfikowanych.

## Funkcjonalność programu

Tworzony system ma charakter narzędzia badawczo-eksperymentalnego wspierającego klasyfikację gatunków muzycznych na podstawie plików MIDI oraz analizę interpretowalności uzyskanych modeli. Funkcjonalności programu obejmują:

* wczytywanie i walidację plików MIDI pochodzących z wybranych datasetów,

* wstępne przetwarzanie danych symbolicznych oraz ich konwersję do wspólnej reprezentacji wykorzystywanej przez modele

* ekstrakcję reprezentacji i cech muzycznych używanych w klasyfikacji oraz analizie interpretowalności

* renowanie modeli bazowych oraz modeli sieci neuronowych do klasyfikacji gatunku muzycznego

* ewaluację modeli z użyciem standardowych metryk klasyfikacyjnych oraz generowanie macierzy pomyłek

* zarządzanie konfiguracją eksperymentów i rejestrowanie ich wyników w sposób reprodukowalny

* zastosowanie metod concept-based interpretability, w szczególności TCAV, do analizy wpływu wybranych konceptów muzycznych na predykcje modelu

* analizę błędnie sklasyfikowanych próbek oraz identyfikację potencjalnych przyczyn błędów

* porównanie wyników uzyskanych na różnych datasetach MIDI, w tym analizę różnic w zakodowanych w nich cechach gatunkowych

* generowanie wykresów, tabel oraz podsumowań wyników na potrzeby dokumentacji i prezentacji projektu
  
  Program jest uruchamiany z poziomu linii poleceń i konfigurowany za pomocą plików konfiguracyjnych, co pozwoliłóna wygodne odtwarzanie eksperymentów oraz porównywanie różnych wariantów modeli oraz ustawień. Część eksperymentów (analiza zbioru danych) realizowano z notatniku Jupyter Notebook.

## Użyte narzędzia

Do przetwarzania plików MIDI zdecydowaliśmy się na wybór biblioteki partitura, ponieważ Foscarin et al. (2022) używa jej w swojej pracy. Ponadto jest ona częściej aktualizowana niż pretty_midi i miditoolkit

### Warstwa badawczo eksperymentalna

* **pandas** - przetwarzanie danych tabelarycznych oraz agregacja wyników eksperymentów

* **partitura** - przetwarzanie plików MIDI oraz ekstrakcja reprezentacji symbolicznych i cech muzycznych wykorzystywanych w dalszej analizie

* **pytorch** - implementacja, trenowanie i ewaluacja modeli sieci neuronowych do klasyfikacji gatunku muzycznego

* **scikit-learn** - analiza wyników, obliczanie metryk ewaluacyjnych oraz implementacja modeli bazowych

* **Captum** -  analiza interpretowalności modelu, w tym zastosowanie metod concept-based interpretability, takich jak TCAV

* **matplotlib** - wizualizacja wyników eksperymentów oraz rezultatów analizy
  interpretowalności

### Warstwa inżyniersko-organizacyjna

* **hydra** - zarządzanie konfiguracją eksperymentów, hiperparametrami oraz wariantami uruchomień

* **Weights & Biases (wanndb)** - monitorowanie eksperymentów oraz porównywanie wyników między uruchomieniami

* **ruff** - statyczna analiza kodu oraz automatyczne formatowanie zgodne z przyjętym stylem projektu

* **uv** - zarządzanie środowiskiem wirtualnym i zależnościami projektu na podstawie
  pliku pyproject.toml

* **make** - Oskryptowane uruchamianie najważniejszych zadań projektu, takich jak instalacja zależności, testy i linting

* **pytest** 

## Użyty dataset

Do eksperymentów wykorzystano otwarty zbiór **XMIDI**. Z to dataset zawierający utwory muzyczne zapisane w formacie MIDI, przeznaczony do analizy symbolicznej muzyki. Dataset został dodatkowo opisany metadanymi, m.in. gatunkiem muzycznym i etykietą emocji, dzięki czemu może być wykorzystany do zadań klasyfikacji muzyki, analizy emocji oraz badania cech charakterystycznych różnych stylów muzycznych. 

Notatnik związany z

Analizowany dataset zawiera próbki muzyczne w formacie MIDI. Każda próbka posiada podstawowe metadane, takie jak gatunek muzyczny, przypisana emocja, identyfikator próbki, nazwa pliku oraz ścieżka do pliku źródłowego.

W zbiorze znajduje się **108 023 próbek**. Dane są kompletne; nie stwierdzono brakujących wartości ani zduplikowanych identyfikatorów `sample_id`. Oznacza to, że dataset nadaje się do dalszej analizy.

Utwory mogą należeć do jednego z 6 gatunków muzyki:

- rock

- pop

- country

- jazz

- classical

- traditional

oraz do jednego z 11 rodzajów emocji: exciting, warm, happy, romantic, funny, sad, angry, lazy itd.

### Rozkład gatunków muzycznych

Analiza rozkładu gatunków wskazuje na to, że dataset nie jest równomiernie zbalansowany. Najwięcej próbek należy do gatunków rock, pop oraz country. Najsłabiej reprezentowany jest gatunek traditional, który stanowi tylko niewielką część całego zbioru.

| Gatunek     | Liczba próbek | Udział w zbiorze [%] |
| ----------- | ------------- | -------------------- |
| rock        | 26 708        | 25                   |
| pop         | 25 582        | 24                   |
| country     | 23 551        | 22                   |
| jazz        | 15 862        | 15                   |
| classical   | 12 660        | 12                   |
| traditional | 3 660         | 3                    |

Procenty zaokrąglono do wartości całkowitych

### Rozkład emocji

Podobne niezbalansowanie występuje również w przypadku etykiet emocji. Najczęściej pojawiającą się emocją jest exciting, natomiast najmniej liczną kategorią jest magnificent.

| Emocja      | Liczba próbek | Udział w zbiorze [%] |
| ----------- | ------------- | -------------------- |
| exciting    | 20 948        | 19                   |
| warm        | 15 090        | 14                   |
| happy       | 13 291        | 12                   |
| romantic    | 12 886        | 12                   |
| funny       | 12 565        | 12                   |
| sad         | 9 038         | 8                    |
| angry       | 8 739         | 8                    |
| lazy        | 4 622         | 4                    |
| quiet       | 4 431         | 4                    |
| fear        | 3 621         | 3                    |
| magnificent | 2 792         | 3                    |

Procenty zaokrąglono do wartości całkowitych. W dalszej cześci zadania nie były analizowane te kategorie gdyż zadanie dotyczyło klasyfikacji gatunku nie emocji związanej z danym utworem.

### Cechy muzyczne próbek

Na podstawie danych MIDI wyznaczono zestaw cech opisujących strukturę muzyczną utworów. Obejmowały one między innymi liczbę nut, czas trwania utworu, gęstość nut, średnią polifonię, wysokości dźwięków, długości nut, wartości velocity oraz odstępy czasowe między zdarzeniami muzycznymi.

Średnio jedna próbka zawierała około 3519 nut, trwała około 176 sekund. Wartości te pokazują, że próbki są dość rozbudowane i zawierają dużą liczbę zdarzeń muzycznych.

Te cechy przekonały nas do wykorzystania tego zbioru w ekspetymentach w ramach realizowanego projektu

### Różnice między gatunkami

Porównanie cech między gatunkami wskazuje, że poszczególne style muzyczne różnią się pod względem struktury symbolicznej. 

- Gatunek **pop** wyróżniał się wysoką liczbą nut i dużą gęstością zdarzeń muzycznych.

- Utwory **country** charakteryzował się relatywnie wysoką średnią polifonią, zaobserwowano wyraźna aktywność rytmiczna

- **Classical** miał wyższą średnią wysokość dźwięków i dłuższe wartości rytmiczne

- Gatunek **traditional** cechował się niższą gęstością nut i spokojniejszym przebiegiem rytmicznym.

- **Jazz** miał większa zmienność wysokości dźwięków

- **Country** miał większa zmienność wysokości dźwięków

Wyniki pokazują, że cechy symboliczne mogą być przydatne w rozróżnianiu gatunków. Jednocześnie różnice nie są na tyle wyraźne, aby całkowicie oddzielić klasy od siebie. Oznacza to, że klasyfikacja gatunku wymaga bardziej złożonego podejścia niż tylko analiza pojedynczych statystyk. 

### Analiza tonalna

W analizie wykorzystano również rozkłady klas wysokości dźwięków, czyli **pitch-class profiles**. Pozwalają one określić, jak często w utworze pojawiają się poszczególne klasy dźwięków niezależnie od oktawy. W takim podejściu dźwięki są sprowadzane do dwunastu klas wysokości odpowiadających dźwiękom skali chromatycznej, niezależnie od oktawy. Oznacza to, że np. wszystkie dźwięki C występujące w różnych oktawach są traktowane jako ta sama klasa wysokości. Analiza pitch-class pozwala określić, które klasy dźwięków występują w utworach najczęściej. Jest to istotne, ponieważ różne gatunki muzyczne mogą wykazywać odmienne preferencje tonalne i harmoniczne.

| Gatunek     | Najsilniejsze klasy wysokości | Interpretacja                                                            |
| ----------- | ----------------------------- | ------------------------------------------------------------------------ |
| classical   | D, G, C, A                    | rozkład stosunkowo równomierny, bez bardzo silnej dominacji jednej klasy |
| country     | F#/Gb, D, C, B                | bardzo wyraźna dominacja klasy F#/Gb                                     |
| jazz        | F#/Gb, C, D, A                | profil tonalny z dominacją F#/Gb, ale mniej skrajny niż w country i pop  |
| pop         | F#/Gb, C, D, A                | silna dominacja F#/Gb                                                    |
| rock        | F#/Gb, D, C, E                | przewaga F#/Gb oraz D i C                                                |
| traditional | D, G, C, A                    | dominują D, G i C; profil zbliżony do classical                          |

Analiza pitch-class wykazała, że poszczególne gatunki różnią się średnim rozkładem klas wysokości dźwięków. W gatunkach country, pop, jazz i rock największy udział miała klasa F#/Gb, natomiast w gatunkach classical i traditional dominowała klasa D. Oznacza to, że gatunki różnią się nie tylko cechami rytmicznymi i statystycznymi, takimi jak liczba nut czy gęstość.

Zaobserwowano, że różne gatunki wykazują odmienne profile tonalne. Oznacza to, że informacja o rozkładzie wysokości dźwięków może być użyteczna jako dodatkowa cecha w klasyfikacji muzyki. Estymowana tonacja została potraktowana jako cecha pomocnicza, ponieważ opisuje ogólne centrum tonalne utworu, ale sama nie wystarcza do jednoznacznego określenia gatunku.

### Reprezenacja piano-roll

Reprezentacja piano-roll pokazuje rozmieszczenie nut w czasie. Oś pozioma odpowiada kolejnym momentom utworu, natomiast oś pionowa reprezentuje wysokości dźwięków. W porównaniu z cechami statystycznymi piano-roll dostarcza bardziej szczegółowej informacji o przebiegu utworu. Przykładowo dwie próbki mogą mieć podobną liczbę nut, podobny czas trwania i zbliżoną gęstość, ale różnić się sposobem rozmieszczenia tych nut w czasie.

W analizowanych przykładach piano-roll pozwolił zauważyć różnice w charakterze przebiegu muzycznego między gatunkami:

| Gatunek     | Obserwacje w piano-roll                                                                              | Interpretacja                                                            |
| ----------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| classical   | większa zmienność wysokości dźwięków, bardziej rozbudowany przebieg melodi, częstsze zmiany rejestru | utwory mają bardziej złożoną strukturę i mniej schematyczny przebieg     |
| jazz        | duża zmienność wysokości, nieregularne układy nut, bardziej swobodny przebieg                        | widoczna większa improwizacyjność i złożoność rytmiczno-melodyczna       |
| pop         | bardziej regularne rozmieszczenie nut, powtarzalne ciągi nut, wyraźniejsze schematy rytmiczne        | struktura jest bardziej uporządkowana i oparta na powtarzalnych motywach |
| rock        | powtarzalne układy, częstsze zwarte grupy nut, stabilniejszy przebieg                                | widoczne są regularne wzorce                                             |
| country     | regularny przebieg, powtarzalne fragmenty, umiarkowana złożoność melodyczna                          | struktura jest uporządkowana, z wyraźnymi schematami                     |
| traditional | mniejsza gęstość nut, wolniejszy przebieg, mniej złożone układy                                      | utwory mają prostszą i bardziej stabilną strukturę czasową               |

W kontekście dalszego modelowania oznacza to, że reprezentacja piano-roll może być użyteczna jako wejście dla modeli analizujących dane sekwencyjne lub obrazowe. Pozwala ona zachować informację o kolejności zdarzeń muzycznych, relacjach czasowych między nutami oraz powtarzalności motywów. Dzięki temu może uzupełniać cechy statystyczne i tonalne, które opisują utwór bardziej ogólnie, ale nie pokazują jego przebiegu w czasie.

### Korelacje i PCA

Analiza korelacji wykazała, że część cech jest ze sobą silnie powiązana. Dotyczy to między innymi liczby nut i gęstości nut, a także cech opisujących długości nut oraz odstępy czasowe między zdarzeniami muzycznymi. Oznacza to, że niektóre zmienne mogą przenosić podobną informację.

Analiza PCA pokazała, że dwie pierwsze składowe główne wyjaśniają jedynie około **26,7% wariancji**. Na wykresie PCA gatunki częściowo się nakładały, co oznacza, że nie da się ich łatwo rozdzielić przy użyciu tylko dwóch głównych wymiarów.

Wyniki PCA potwierdzają, że problem klasyfikacji gatunków jest złożony. **Cechy statystyczne dostarczają użytecznych informacji, ale same nie pozwalają na jednoznaczne rozdzielenie wszystkich gatunków.**

## Implementacja modeli

## Błędy modeli

## Ewaluacja modeli

## Interperowalność TACV

## Zużyte zasoby

## Wnioski

* d








