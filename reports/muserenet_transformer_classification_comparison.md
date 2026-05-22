# Porównanie raportów klasyfikacyjnych: MuSeReNet vs MIDI Transformer

Źródła:

- `slurm/logs/xmidi-muserenet-2595741.err`
- `slurm/logs/xmidi-transformer-2595030.err`

Raporty dotyczą walidacji na tym samym zbiorze `10 485` przykładów. Mapowanie klas wynika z sortowania nazw gatunków w `build_label_mapping`: `0=classical`, `1=country`, `2=jazz`, `3=pop`, `4=rock`, `5=traditional`.

## Podsumowanie

| Metryka | MuSeReNet | MIDI Transformer | Różnica |
| --- | ---: | ---: | ---: |
| Accuracy | 0.53 | 0.46 | +0.07 |
| Macro precision | 0.50 | 0.40 | +0.10 |
| Macro recall | 0.48 | 0.39 | +0.09 |
| Macro F1 | 0.48 | 0.39 | +0.09 |
| Weighted precision | 0.53 | 0.45 | +0.08 |
| Weighted recall | 0.53 | 0.46 | +0.07 |
| Weighted F1 | 0.52 | 0.45 | +0.07 |

## Porównanie per klasa

| Klasa | Gatunek | Support | MuSeReNet P | MuSeReNet R | MuSeReNet F1 | Transformer P | Transformer R | Transformer F1 | Różnica F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | classical | 1 209 | 0.54 | 0.74 | 0.62 | 0.62 | 0.63 | 0.63 | -0.01 |
| 1 | country | 2 308 | 0.64 | 0.46 | 0.54 | 0.51 | 0.47 | 0.49 | +0.05 |
| 2 | jazz | 1 540 | 0.51 | 0.52 | 0.52 | 0.41 | 0.19 | 0.26 | +0.26 |
| 3 | pop | 2 476 | 0.50 | 0.53 | 0.52 | 0.40 | 0.59 | 0.48 | +0.04 |
| 4 | rock | 2 602 | 0.50 | 0.56 | 0.53 | 0.46 | 0.48 | 0.47 | +0.06 |
| 5 | traditional | 350 | 0.29 | 0.10 | 0.15 | 0.00 | 0.00 | 0.00 | +0.15 |

## Najważniejsze obserwacje

- MuSeReNet jest wyraźnie mocniejszym baseline'em: ma wyższe `accuracy`, `macro F1` i `weighted F1`.
- Największa różnica jest dla klasy `jazz`: MuSeReNet osiąga `F1=0.52`, a Transformer tylko `F1=0.26`.
- Transformer minimalnie wygrywa tylko dla klasy `classical` (`F1=0.63` vs `0.62`).
- Oba modele mają problem z klasą `traditional`, która ma najmniejsze wsparcie w walidacji (`350` przykładów). Transformer w ogóle jej nie rozpoznaje (`F1=0.00`), a MuSeReNet osiąga tylko `F1=0.15`.
- MuSeReNet ma wyraźny gap między treningiem i walidacją w końcowych epokach, więc wynik jest lepszy od Transformera, ale nadal sugeruje overfitting.
