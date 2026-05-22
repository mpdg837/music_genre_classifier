# Standup
## Modele

- Dodano spójny pipeline trenowania modeli neuronowych oparty o Hydrę:
  `configs/neural_config.yaml`, `scripts/train_neural.py` oraz konfiguracje modeli w
  `configs/model/`.
- Dodano dwa klasyczne baseline'y neuronowe dla danych MIDI:
  - **MuSeReNet** jako wielorozdzielczy model CNN operujący na piano-rollu,
  - **MIDI Transformer** jako enkoder Transformer operujący na sekwencji nut z maską paddingu.
- Rozszerzono warstwę danych w `midi_xai/data/create_dataset.py`:
  - generowanie piano-rolli na żądanie,
  - generowanie macierzy nut dla Transformera,
  - ekstrakcję cech statystycznych nut używanych także później do konceptów TCAV.
- Dodano integrację **MusicBERT** przez Hugging Face:
  - dataset/tokenizację REMI+BPE w `midi_xai/data/musicbert_dataset.py`,
  - model klasyfikacyjny `MusicBertGenreClassifier`,
  - konfigurację pełnego fine-tuningu oraz wariant z zamrożonym enkoderem.
- Dodano wariant **MusicBERT frozen head**: zamrożony enkoder MusicBERT + większa,
  nazwana głowica klasyfikacyjna `[1024, 512]`, lepiej nadająca się do późniejszej analizy TCAV.
- Dodano harmonogram uczenia z warmupem, gradient accumulation, class weighting,
  ładowanie checkpointów oraz oddzielne ścieżki zapisu wag, aby nie nadpisywać starszych wyników.
- Dodano pomocniczy skrypt `scripts/evaluate_musicbert_embeddings.py` do oceny zamrożonych
  embeddingów MusicBERT z prostą głowicą klasyfikacyjną.
- Rozszerzono `Makefile`, `README.md`, testy zależności i smoke-testy modeli.
- Dodano skrypty Slurm dla uruchamiania treningu MuSeReNet, Transformera, MusicBERT oraz
  wariantu MusicBERT frozen-head na klastrze.

## Interpretowalność

- Dodano osobny pipeline interpretowalności oparty o **TCAV** dla baseline'u MuSeReNet.
- Implementacja została uporządkowana w trzech plikach:
  - `concepts.py` - manifesty konceptów, dataset konceptów, losowe kontrole,
  - `core.py` - integracja z Captum, ładowanie checkpointów, scoring TCAV,
  - `reports.py` - zapis wyników i podsumowania istotności.
- TCAV korzysta z biblioteki **Captum**: budowane są obiekty `Concept`, trenowane są CAV-y
  i wywoływane jest `TCAV.interpret(...)` dla klas gatunków muzycznych.
- Dodano własny wrapper liniowego klasyfikatora CAV opartego o `LinearSVC`, kompatybilny
  z API Captum.
- Dodano konfigurację `configs/tcav/muserenet_baseline.yaml` oraz główny entrypoint
  `configs/tcav_config.yaml`.
- Dodano automatyczne przygotowanie konceptów z cech symbolicznych MIDI, m.in.:
  `high_note_density`, `low_note_density`, `high_polyphony`, `wide_pitch_range`,
  `high/low_pitch_register`, `strong_velocity`, `long_notes`, `short_notes`,
  `irregular_ioi`, `regular_ioi`.
- Dodano przygotowanie losowych zbiorów kontrolnych dla TCAV.
- Dodano skrypty:
  - `scripts/prepare_tcav_concepts.py`,
  - `scripts/prepare_tcav_controls.py`,
  - `scripts/run_tcav_muserenet.py`.
- Wyniki TCAV są zapisywane jako:
  - cache CAV Captum,
  - `tcav_scores.csv`,
  - `tcav_summary.csv`,
  - `run_metadata.json`.
