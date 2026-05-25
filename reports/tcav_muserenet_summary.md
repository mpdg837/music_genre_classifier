# Wyniki TCAV dla MuSeReNet

Źródło: `/net/tscratch/people/plgatarsander/WIMU_DATA/tcav/muserenet_baseline/scores/tcav_summary.csv`

Raport zawiera `132` testy TCAV: `11` konceptów × `2` warstwy × `6` klas. Istotność liczona była względem skorygowanego progu `corrected_alpha = 0.0003787878787878788`, po `10` próbach kontrolnych na test.

## Podsumowanie

| Liczba testów | Istotne | Pozytywne | Negatywne |
| ---: | ---: | ---: | ---: |
| 132 | 20 | 10 | 10 |

Wszystkie istotne wyniki pojawiły się dla warstwy `classifier.1`. Dla `classifier.2` nie ma wyników istotnych po korekcie wielokrotnych testów.

## Wyniki istotne statystycznie

| Koncept | Warstwa | Klasa | TCAV mean | TCAV std | p-value | Kierunek |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `high_note_density` | `classifier.1` | country | 0.271 | 0.076 | 5.02e-06 | negative |
| `high_note_density` | `classifier.1` | jazz | 0.191 | 0.110 | 9.65e-06 | negative |
| `high_note_density` | `classifier.1` | pop | 0.839 | 0.064 | 4.07e-08 | positive |
| `high_pitch_register` | `classifier.1` | classical | 0.938 | 0.043 | 1.37e-10 | positive |
| `high_pitch_register` | `classifier.1` | jazz | 0.189 | 0.134 | 4.31e-05 | negative |
| `high_pitch_register` | `classifier.1` | pop | 0.221 | 0.121 | 4.50e-05 | negative |
| `high_pitch_register` | `classifier.1` | traditional | 0.791 | 0.107 | 1.28e-05 | positive |
| `high_polyphony` | `classifier.1` | rock | 0.261 | 0.122 | 1.55e-04 | negative |
| `irregular_ioi` | `classifier.1` | classical | 0.953 | 0.039 | 4.16e-11 | positive |
| `long_notes` | `classifier.1` | classical | 0.952 | 0.039 | 4.28e-11 | positive |
| `long_notes` | `classifier.1` | pop | 0.221 | 0.092 | 5.19e-06 | negative |
| `long_notes` | `classifier.1` | rock | 0.214 | 0.107 | 1.36e-05 | negative |
| `long_notes` | `classifier.1` | traditional | 0.721 | 0.104 | 8.49e-05 | positive |
| `low_note_density` | `classifier.1` | classical | 0.898 | 0.098 | 4.14e-07 | positive |
| `low_note_density` | `classifier.1` | jazz | 0.150 | 0.100 | 1.53e-06 | negative |
| `low_pitch_register` | `classifier.1` | country | 0.687 | 0.087 | 7.65e-05 | positive |
| `regular_ioi` | `classifier.1` | country | 0.288 | 0.110 | 1.77e-04 | negative |
| `short_notes` | `classifier.1` | jazz | 0.174 | 0.133 | 2.96e-05 | negative |
| `short_notes` | `classifier.1` | rock | 0.744 | 0.108 | 5.50e-05 | positive |
| `strong_velocity` | `classifier.1` | rock | 0.692 | 0.098 | 1.64e-04 | positive |

## Pełna tabela wyników

| Koncept | Warstwa | Klasa | TCAV mean | TCAV std | p-value | Istotne | Kierunek | Próby |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| `high_note_density` | `classifier.1` | classical | 0.307 | 0.123 | 7.89e-04 | nie | not_significant | 10 |
| `high_note_density` | `classifier.1` | country | 0.271 | 0.076 | 5.02e-06 | tak | negative | 10 |
| `high_note_density` | `classifier.1` | jazz | 0.191 | 0.110 | 9.65e-06 | tak | negative | 10 |
| `high_note_density` | `classifier.1` | pop | 0.839 | 0.064 | 4.07e-08 | tak | positive | 10 |
| `high_note_density` | `classifier.1` | rock | 0.347 | 0.098 | 7.75e-04 | nie | not_significant | 10 |
| `high_note_density` | `classifier.1` | traditional | 0.595 | 0.123 | 3.68e-02 | nie | not_significant | 10 |
| `high_note_density` | `classifier.2` | classical | 0.100 | 0.316 | - | nie | not_significant | 10 |
| `high_note_density` | `classifier.2` | country | 0.400 | 0.516 | 5.55e-01 | nie | not_significant | 10 |
| `high_note_density` | `classifier.2` | jazz | 0.000 | 0.000 | - | nie | not_significant | 10 |
| `high_note_density` | `classifier.2` | pop | 0.800 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `high_note_density` | `classifier.2` | rock | 0.200 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `high_note_density` | `classifier.2` | traditional | 0.400 | 0.516 | 5.55e-01 | nie | not_significant | 10 |
| `high_pitch_register` | `classifier.1` | classical | 0.938 | 0.043 | 1.37e-10 | tak | positive | 10 |
| `high_pitch_register` | `classifier.1` | country | 0.506 | 0.147 | 9.03e-01 | nie | not_significant | 10 |
| `high_pitch_register` | `classifier.1` | jazz | 0.189 | 0.134 | 4.31e-05 | tak | negative | 10 |
| `high_pitch_register` | `classifier.1` | pop | 0.221 | 0.121 | 4.50e-05 | tak | negative | 10 |
| `high_pitch_register` | `classifier.1` | rock | 0.379 | 0.193 | 7.89e-02 | nie | not_significant | 10 |
| `high_pitch_register` | `classifier.1` | traditional | 0.791 | 0.107 | 1.28e-05 | tak | positive | 10 |
| `high_pitch_register` | `classifier.2` | classical | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `high_pitch_register` | `classifier.2` | country | 0.200 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `high_pitch_register` | `classifier.2` | jazz | 0.300 | 0.483 | 2.23e-01 | nie | not_significant | 10 |
| `high_pitch_register` | `classifier.2` | pop | 0.000 | 0.000 | - | nie | not_significant | 10 |
| `high_pitch_register` | `classifier.2` | rock | 0.500 | 0.527 | 1.00e+00 | nie | not_significant | 10 |
| `high_pitch_register` | `classifier.2` | traditional | 0.800 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `high_polyphony` | `classifier.1` | classical | 0.473 | 0.229 | 7.22e-01 | nie | not_significant | 10 |
| `high_polyphony` | `classifier.1` | country | 0.499 | 0.206 | 9.93e-01 | nie | not_significant | 10 |
| `high_polyphony` | `classifier.1` | jazz | 0.682 | 0.167 | 7.32e-03 | nie | not_significant | 10 |
| `high_polyphony` | `classifier.1` | pop | 0.475 | 0.149 | 6.04e-01 | nie | not_significant | 10 |
| `high_polyphony` | `classifier.1` | rock | 0.261 | 0.122 | 1.55e-04 | tak | negative | 10 |
| `high_polyphony` | `classifier.1` | traditional | 0.457 | 0.149 | 3.86e-01 | nie | not_significant | 10 |
| `high_polyphony` | `classifier.2` | classical | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `high_polyphony` | `classifier.2` | country | 0.900 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `high_polyphony` | `classifier.2` | jazz | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `high_polyphony` | `classifier.2` | pop | 0.500 | 0.527 | 1.00e+00 | nie | not_significant | 10 |
| `high_polyphony` | `classifier.2` | rock | 0.100 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `high_polyphony` | `classifier.2` | traditional | 0.800 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.1` | classical | 0.953 | 0.039 | 4.16e-11 | tak | positive | 10 |
| `irregular_ioi` | `classifier.1` | country | 0.466 | 0.140 | 4.68e-01 | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.1` | jazz | 0.250 | 0.192 | 2.67e-03 | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.1` | pop | 0.269 | 0.135 | 4.26e-04 | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.1` | rock | 0.543 | 0.175 | 4.61e-01 | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.1` | traditional | 0.633 | 0.123 | 7.67e-03 | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.2` | classical | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.2` | country | 0.900 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.2` | jazz | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.2` | pop | 0.400 | 0.516 | 5.55e-01 | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.2` | rock | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `irregular_ioi` | `classifier.2` | traditional | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `long_notes` | `classifier.1` | classical | 0.952 | 0.039 | 4.28e-11 | tak | positive | 10 |
| `long_notes` | `classifier.1` | country | 0.404 | 0.145 | 6.70e-02 | nie | not_significant | 10 |
| `long_notes` | `classifier.1` | jazz | 0.465 | 0.193 | 5.76e-01 | nie | not_significant | 10 |
| `long_notes` | `classifier.1` | pop | 0.221 | 0.092 | 5.19e-06 | tak | negative | 10 |
| `long_notes` | `classifier.1` | rock | 0.214 | 0.107 | 1.36e-05 | tak | negative | 10 |
| `long_notes` | `classifier.1` | traditional | 0.721 | 0.104 | 8.49e-05 | tak | positive | 10 |
| `long_notes` | `classifier.2` | classical | 0.900 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `long_notes` | `classifier.2` | country | 0.300 | 0.483 | 2.23e-01 | nie | not_significant | 10 |
| `long_notes` | `classifier.2` | jazz | 0.300 | 0.483 | 2.23e-01 | nie | not_significant | 10 |
| `long_notes` | `classifier.2` | pop | 0.400 | 0.516 | 5.55e-01 | nie | not_significant | 10 |
| `long_notes` | `classifier.2` | rock | 0.200 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `long_notes` | `classifier.2` | traditional | 0.900 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `low_note_density` | `classifier.1` | classical | 0.898 | 0.098 | 4.14e-07 | tak | positive | 10 |
| `low_note_density` | `classifier.1` | country | 0.571 | 0.145 | 1.53e-01 | nie | not_significant | 10 |
| `low_note_density` | `classifier.1` | jazz | 0.150 | 0.100 | 1.53e-06 | tak | negative | 10 |
| `low_note_density` | `classifier.1` | pop | 0.260 | 0.197 | 3.92e-03 | nie | not_significant | 10 |
| `low_note_density` | `classifier.1` | rock | 0.595 | 0.189 | 1.48e-01 | nie | not_significant | 10 |
| `low_note_density` | `classifier.1` | traditional | 0.623 | 0.086 | 1.47e-03 | nie | not_significant | 10 |
| `low_note_density` | `classifier.2` | classical | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `low_note_density` | `classifier.2` | country | 0.900 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `low_note_density` | `classifier.2` | jazz | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `low_note_density` | `classifier.2` | pop | 0.300 | 0.483 | 2.23e-01 | nie | not_significant | 10 |
| `low_note_density` | `classifier.2` | rock | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `low_note_density` | `classifier.2` | traditional | 1.000 | 0.000 | - | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.1` | classical | 0.772 | 0.200 | 1.97e-03 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.1` | country | 0.687 | 0.087 | 7.65e-05 | tak | positive | 10 |
| `low_pitch_register` | `classifier.1` | jazz | 0.307 | 0.139 | 1.76e-03 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.1` | pop | 0.282 | 0.183 | 4.40e-03 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.1` | rock | 0.524 | 0.134 | 5.86e-01 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.1` | traditional | 0.316 | 0.188 | 1.29e-02 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.2` | classical | 0.800 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.2` | country | 0.600 | 0.516 | 5.55e-01 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.2` | jazz | 0.100 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.2` | pop | 0.100 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.2` | rock | 0.600 | 0.516 | 5.55e-01 | nie | not_significant | 10 |
| `low_pitch_register` | `classifier.2` | traditional | 0.000 | 0.000 | - | nie | not_significant | 10 |
| `regular_ioi` | `classifier.1` | classical | 0.253 | 0.150 | 5.52e-04 | nie | not_significant | 10 |
| `regular_ioi` | `classifier.1` | country | 0.288 | 0.110 | 1.77e-04 | tak | negative | 10 |
| `regular_ioi` | `classifier.1` | jazz | 0.220 | 0.161 | 3.79e-04 | nie | not_significant | 10 |
| `regular_ioi` | `classifier.1` | pop | 0.540 | 0.129 | 3.50e-01 | nie | not_significant | 10 |
| `regular_ioi` | `classifier.1` | rock | 0.633 | 0.150 | 2.08e-02 | nie | not_significant | 10 |
| `regular_ioi` | `classifier.1` | traditional | 0.676 | 0.107 | 5.85e-04 | nie | not_significant | 10 |
| `regular_ioi` | `classifier.2` | classical | 0.200 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `regular_ioi` | `classifier.2` | country | 0.100 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `regular_ioi` | `classifier.2` | jazz | 0.200 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `regular_ioi` | `classifier.2` | pop | 0.900 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `regular_ioi` | `classifier.2` | rock | 0.000 | 0.000 | - | nie | not_significant | 10 |
| `regular_ioi` | `classifier.2` | traditional | 0.200 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `short_notes` | `classifier.1` | classical | 0.326 | 0.108 | 6.70e-04 | nie | not_significant | 10 |
| `short_notes` | `classifier.1` | country | 0.308 | 0.149 | 2.73e-03 | nie | not_significant | 10 |
| `short_notes` | `classifier.1` | jazz | 0.174 | 0.133 | 2.96e-05 | tak | negative | 10 |
| `short_notes` | `classifier.1` | pop | 0.546 | 0.154 | 3.70e-01 | nie | not_significant | 10 |
| `short_notes` | `classifier.1` | rock | 0.744 | 0.108 | 5.50e-05 | tak | positive | 10 |
| `short_notes` | `classifier.1` | traditional | 0.316 | 0.114 | 6.26e-04 | nie | not_significant | 10 |
| `short_notes` | `classifier.2` | classical | 0.700 | 0.483 | 2.23e-01 | nie | not_significant | 10 |
| `short_notes` | `classifier.2` | country | 0.700 | 0.483 | 2.23e-01 | nie | not_significant | 10 |
| `short_notes` | `classifier.2` | jazz | 0.500 | 0.527 | 1.00e+00 | nie | not_significant | 10 |
| `short_notes` | `classifier.2` | pop | 0.800 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `short_notes` | `classifier.2` | rock | 0.700 | 0.483 | 2.23e-01 | nie | not_significant | 10 |
| `short_notes` | `classifier.2` | traditional | 0.600 | 0.516 | 5.55e-01 | nie | not_significant | 10 |
| `strong_velocity` | `classifier.1` | classical | 0.543 | 0.195 | 5.07e-01 | nie | not_significant | 10 |
| `strong_velocity` | `classifier.1` | country | 0.487 | 0.051 | 4.60e-01 | nie | not_significant | 10 |
| `strong_velocity` | `classifier.1` | jazz | 0.385 | 0.091 | 3.18e-03 | nie | not_significant | 10 |
| `strong_velocity` | `classifier.1` | pop | 0.563 | 0.167 | 2.61e-01 | nie | not_significant | 10 |
| `strong_velocity` | `classifier.1` | rock | 0.692 | 0.098 | 1.64e-04 | tak | positive | 10 |
| `strong_velocity` | `classifier.1` | traditional | 0.569 | 0.170 | 2.35e-01 | nie | not_significant | 10 |
| `strong_velocity` | `classifier.2` | classical | 0.100 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `strong_velocity` | `classifier.2` | country | 0.100 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `strong_velocity` | `classifier.2` | jazz | 0.000 | 0.000 | - | nie | not_significant | 10 |
| `strong_velocity` | `classifier.2` | pop | 0.000 | 0.000 | - | nie | not_significant | 10 |
| `strong_velocity` | `classifier.2` | rock | 0.400 | 0.516 | 5.55e-01 | nie | not_significant | 10 |
| `strong_velocity` | `classifier.2` | traditional | 0.100 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.1` | classical | 0.288 | 0.185 | 5.52e-03 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.1` | country | 0.648 | 0.187 | 3.30e-02 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.1` | jazz | 0.278 | 0.165 | 2.11e-03 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.1` | pop | 0.683 | 0.150 | 3.96e-03 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.1` | rock | 0.551 | 0.192 | 4.20e-01 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.1` | traditional | 0.524 | 0.179 | 6.84e-01 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.2` | classical | 0.500 | 0.527 | 1.00e+00 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.2` | country | 0.100 | 0.316 | 3.11e-03 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.2` | jazz | 0.300 | 0.483 | 2.23e-01 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.2` | pop | 0.800 | 0.422 | 5.10e-02 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.2` | rock | 0.500 | 0.527 | 1.00e+00 | nie | not_significant | 10 |
| `wide_pitch_range` | `classifier.2` | traditional | 0.300 | 0.483 | 2.23e-01 | nie | not_significant | 10 |
