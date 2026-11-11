# Análise x3 — bateria triplicada vs números do paper

Relatório gerado por análise offline (sem tocar em código do projeto).

## 0. Fontes, casamento e método

| bateria | diretório | arquivos | seeds | uso |
|---|---|---|---|---|
| antiga (paper) | `runs/` | 1855 (1851 com métrica) | t1: 0–4, t2/t3/t4: 0–2 | números publicados |
| SNN corrigida | `runs_snn_fixed/` | 440 | t1: 0–4, t2/t3: 0–2 | referência de receita (ver §4.3) |
| nova x3 | `runs_x3/` | 1500 | t1: 0–14, t2: 0–8 | CNN-LSTM + SNN |
| nova x3 SSM | `runs_ssm_x3/` | 1065 | t1: 0–14, t2/t3tau: 0–8 | Mamba/SSM |

- Casamento run↔manifesto feito por `config.manifest.run_id` (o `config_hash` das linhas
  de seed expandida nos manifestos x3 está stale — repete o hash da seed-template; ver §4.5).
- Estatísticas usam o **config gravado no resultado** (`config.manifest.config`), não o manifesto.
- `final.diverged=true` (macro-F1=0 marcador) é excluído das médias e contado à parte.
- FRR/admissão recomputados de `rounds[].consensus.rejected` × `clients[].attack`,
  mesma lógica de `experiments/report.py::Row.gate_stats`.
- Ambiente mudou entre baterias: antiga = mix RTX 3060/3070/3090/4080, torch 2.5–2.13;
  x3 = 100% `py3.12.14 | torch 2.9.1+cu128 | RTX 5090`. `git_commit` = `unknown` em todos os x3.

Cobertura: a x3 recobre **233 dos 511 templates** da bateria antiga. NÃO recobre: `temporal-*`
(78 templates — split cronológico), `t4mag` (48 — magnitude sweep), `t3tau` para CNN-LSTM/SNN
(144; SSM ganhou t3tau novo com 9 seeds, mas só a variante de fonte `t0p3` do grid t2c) e
8 variações `t2e` (`codingrate-ts4/ts16` para SNN, `ts4/ts16` para SSM).

## 1. Tabela comparativa por template (macro-F1)

`old` = `runs/` (n=5 em t1, n=3 em t2/t3); `new` = `runs_x3`∪`runs_ssm_x3` (n=15 em t1,
n=9 em t2/t3tau). Δ = média_nova − média_antiga. **⚑** = |Δ| > 0.05.
`†` = divergence por mudança de recipe (ver §4.3). `·div` = runs divergidos excluídos.

### bloco `t1a`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `t1a-centralized-cnn_lstm` | 5 | 0.666±0.053 | 15 | 0.582±0.100 | -0.084 | ⚑† |

| `t1a-centralized-snn` | 5 | 0.700±0.035 | 15 | 0.586±0.085 | -0.115 | ⚑† |

| `t1a-centralized-ssm` | 5 | 0.916±0.025 | 15 | 0.849±0.049 | -0.068 | ⚑† |

| `t1a-fedavg-cnn_lstm` | 5 | 0.620±0.071 | 15 | 0.600±0.048 | -0.021 | |

| `t1a-fedavg-snn` | 5 | 0.161±0.012 | 15 | 0.161±0.021 | -0.000 | |

| `t1a-fedavg-ssm` | 5 | 0.758±0.047 | 15 | 0.770±0.037 | +0.013 | |

| `t1a-fedprox-cnn_lstm` | 5 | 0.622±0.069 | 15 | 0.603±0.056 | -0.019 | |

| `t1a-fedprox-snn` | 5 | 0.163±0.015 | 15 | 0.163±0.020 | +0.001 | |

| `t1a-fedprox-ssm` | 5 | 0.747±0.032 | 15 | 0.751±0.030 | +0.003 | |

| `t1a-local-cnn_lstm` | 5 | 0.298±0.020 | 15 | 0.292±0.028 | -0.006 | |

| `t1a-local-snn` | 5 | 0.130±0.004 | 15 | 0.129±0.008 | -0.001 | |

| `t1a-local-ssm` | 5 | 0.202±0.009 | 15 | 0.197±0.013 | -0.005 | |

| `t1a-pipeline_geomed-cnn_lstm` | 5 | 0.475±0.070 | 15 | 0.454±0.067 | -0.021 | |

| `t1a-pipeline_geomed-snn` | 5 | 0.140±0.014 | 15 | 0.136±0.019 | -0.004 | |

| `t1a-pipeline_geomed-ssm` | 5 | 0.138±0.012 | 15 | 0.156±0.048 | +0.018 | |

| `t1a-pipeline_krum-cnn_lstm` | 5 | 0.495±0.039 | 15 | 0.476±0.069 | -0.018 | |

| `t1a-pipeline_krum-snn` | 5 | 0.125±0.020 | 15 | 0.126±0.018 | +0.000 | |

| `t1a-pipeline_krum-ssm` | 5 | 0.172±0.043 | 15 | 0.198±0.074 | +0.026 | |

### bloco `t1b`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `t1b-geomed_alone-cnn_lstm-atkadaptive_cosine` | 5 | 0.631±0.063 | 15 | 0.609±0.070 | -0.022 | |

| `t1b-geomed_alone-cnn_lstm-atkgaussian_noise` | 5 | 0.631±0.057 | 15 | 0.587±0.043 | -0.044 | |

| `t1b-geomed_alone-cnn_lstm-atklabel_flip` | 5 | 0.525±0.094 | 15 | 0.544±0.069 | +0.019 | |

| `t1b-geomed_alone-cnn_lstm-atksign_flip` | 5 | 0.428±0.065 | 15 | 0.437±0.060 | +0.009 | |

| `t1b-geomed_alone-snn-atkadaptive_cosine` | 5 | 0.151±0.011 | 15 | 0.149±0.018 | -0.002 | |

| `t1b-geomed_alone-snn-atkgaussian_noise` | 5 | 0.148±0.012 | 15 | 0.151±0.017 | +0.003 | |

| `t1b-geomed_alone-snn-atklabel_flip` | 5 | 0.093±0.026 | 15 | 0.103±0.025 | +0.010 | |

| `t1b-geomed_alone-snn-atksign_flip` | 5 | 0.136±0.013 | 15 | 0.132±0.013 | -0.003 | |

| `t1b-geomed_alone-ssm-atkadaptive_cosine` | 5 | 0.710±0.053 | 15 | 0.726±0.037 | +0.016 | |

| `t1b-geomed_alone-ssm-atkgaussian_noise` | 5 | 0.662±0.082 | 15 | 0.692±0.060 | +0.030 | |

| `t1b-geomed_alone-ssm-atklabel_flip` | 5 | 0.413±0.017 | 15 | 0.439±0.039 | +0.026 | |

| `t1b-geomed_alone-ssm-atksign_flip` | 5 | 0.359±0.077 | 15 | 0.389±0.071 | +0.029 | |

| `t1b-krum_alone-cnn_lstm-atkadaptive_cosine` | 5 | 0.121±0.068 | 15 | 0.100±0.050 | -0.021 | |

| `t1b-krum_alone-cnn_lstm-atkgaussian_noise` | 5 | 0.371±0.096 | 15 | 0.408±0.101 | +0.037 | |

| `t1b-krum_alone-cnn_lstm-atklabel_flip` | 5 | 0.396±0.083 | 15 | 0.359±0.074 | -0.037 | |

| `t1b-krum_alone-cnn_lstm-atksign_flip` | 5 | 0.363±0.135 | 15 | 0.355±0.095 | -0.008 | |

| `t1b-krum_alone-snn-atkadaptive_cosine` | 5 | 0.095±0.051 | 15 | 0.060±0.062 | -0.036 | |

| `t1b-krum_alone-snn-atkgaussian_noise` | 5 | 0.143±0.012 | 15 | 0.137±0.024 | -0.006 | |

| `t1b-krum_alone-snn-atklabel_flip` | 5 | 0.131±0.027 | 15 | 0.124±0.024 | -0.007 | |

| `t1b-krum_alone-snn-atksign_flip` | 5 | 0.144±0.012 | 15 | 0.128±0.020 | -0.016 | |

| `t1b-krum_alone-ssm-atkadaptive_cosine` | 5 | 0.058±0.028 | 15 | 0.043±0.033 | -0.015 | |

| `t1b-krum_alone-ssm-atkgaussian_noise` | 5 | 0.276±0.038 | 15 | 0.261±0.042 | -0.015 | |

| `t1b-krum_alone-ssm-atklabel_flip` | 5 | 0.299±0.031 | 15 | 0.245±0.052 | -0.054 | ⚑ |

| `t1b-krum_alone-ssm-atksign_flip` | 5 | 0.247±0.051 | 15 | 0.224±0.060 | -0.023 | |

| `t1b-no_defense-cnn_lstm-atkadaptive_cosine` | 5 | 0.621±0.059 | 15 | 0.629±0.057 | +0.008 | |

| `t1b-no_defense-cnn_lstm-atkgaussian_noise` | 5 | 0.053±0.017 | 15 | 0.059±0.017 | +0.006 | |

| `t1b-no_defense-cnn_lstm-atklabel_flip` | 5 | 0.495±0.067 | 15 | 0.509±0.069 | +0.014 | |

| `t1b-no_defense-cnn_lstm-atksign_flip` | 5 | 0.366±0.067 | 15 | 0.392±0.063 | +0.026 | |

| `t1b-no_defense-snn-atkadaptive_cosine` | 5 | 0.147±0.009 | 15 | 0.152±0.018 | +0.006 | |

| `t1b-no_defense-snn-atkgaussian_noise` | 5 | 0.117±0.009 | 15 | 0.100±0.029 | -0.017 | |

| `t1b-no_defense-snn-atklabel_flip` | 5 | 0.083±0.024 | 15 | 0.093±0.028 | +0.011 | |

| `t1b-no_defense-snn-atksign_flip` | 5 | 0.125±0.008 | 15 | 0.124±0.012 | -0.001 | |

| `t1b-no_defense-ssm-atkadaptive_cosine` | 5 | 0.714±0.070 | 15 | 0.739±0.054 | +0.025 | |

| `t1b-no_defense-ssm-atkgaussian_noise` | 5 | 0.074±0.016 | 15 | 0.076±0.010 | +0.002 | |

| `t1b-no_defense-ssm-atklabel_flip` | 5 | 0.406±0.014 | 15 | 0.418±0.036 | +0.012 | |

| `t1b-no_defense-ssm-atksign_flip` | 5 | 0.285±0.049 | 15 | 0.293±0.059 | +0.009 | |

| `t1b-pipeline_geomed-cnn_lstm-atkadaptive_cosine` | 5 | 0.413±0.180 | 15 | 0.420±0.140 | +0.007 | |

| `t1b-pipeline_geomed-cnn_lstm-atkgaussian_noise` | 5 | 0.411±0.068 | 15 | 0.377±0.056 | -0.034 | |

| `t1b-pipeline_geomed-cnn_lstm-atklabel_flip` | 5 | 0.396±0.044 | 15 | 0.357±0.132 | -0.040 | |

| `t1b-pipeline_geomed-cnn_lstm-atksign_flip` | 5 | 0.312±0.079 | 15 | 0.320±0.051 | +0.009 | |

| `t1b-pipeline_geomed-snn-atkadaptive_cosine` | 5 | 0.085±0.051 | 15 | 0.082±0.048 | -0.004 | |

| `t1b-pipeline_geomed-snn-atkgaussian_noise` | 5 | 0.114±0.009 | 15 | 0.106±0.028 | -0.008 | |

| `t1b-pipeline_geomed-snn-atklabel_flip` | 5 | 0.109±0.014 | 15 | 0.091±0.031 | -0.018 | |

| `t1b-pipeline_geomed-snn-atksign_flip` | 5 | 0.113±0.007 | 15 | 0.097±0.039 | -0.016 | |

| `t1b-pipeline_geomed-ssm-atkadaptive_cosine` | 5 | 0.064±0.025 | 15 | 0.051±0.030 | -0.013 | |

| `t1b-pipeline_geomed-ssm-atkgaussian_noise` | 5 | 0.079±0.063 | 15 | 0.084±0.050 | +0.005 | |

| `t1b-pipeline_geomed-ssm-atklabel_flip` | 5 | 0.072±0.062 | 15 | 0.095±0.049 | +0.023 | |

| `t1b-pipeline_geomed-ssm-atksign_flip` | 5 | 0.050±0.049 | 15 | 0.076±0.048 | +0.027 | |

| `t1b-pipeline_krum-cnn_lstm-atkadaptive_cosine` | 5 | 0.216±0.137 | 15 | 0.235±0.088 | +0.019 | |

| `t1b-pipeline_krum-cnn_lstm-atkgaussian_noise` | 5 | 0.355±0.075 | 15 | 0.366±0.076 | +0.011 | |

| `t1b-pipeline_krum-cnn_lstm-atklabel_flip` | 5 | 0.287±0.173 | 15 | 0.368±0.065 | +0.082 | ⚑ |

| `t1b-pipeline_krum-cnn_lstm-atksign_flip` | 5 | 0.332±0.056 | 15 | 0.323±0.069 | -0.009 | |

| `t1b-pipeline_krum-snn-atkadaptive_cosine` | 5 | 0.118±0.009 | 15 | 0.095±0.036 | -0.022 | |

| `t1b-pipeline_krum-snn-atkgaussian_noise` | 5 | 0.116±0.007 | 15 | 0.106±0.029 | -0.010 | |

| `t1b-pipeline_krum-snn-atklabel_flip` | 5 | 0.112±0.012 | 15 | 0.095±0.042 | -0.016 | |

| `t1b-pipeline_krum-snn-atksign_flip` | 5 | 0.109±0.008 | 15 | 0.097±0.039 | -0.012 | |

| `t1b-pipeline_krum-ssm-atkadaptive_cosine` | 5 | 0.041±0.036 | 15 | 0.031±0.025 | -0.010 | |

| `t1b-pipeline_krum-ssm-atkgaussian_noise` | 5 | 0.076±0.059 | 15 | 0.089±0.057 | +0.013 | |

| `t1b-pipeline_krum-ssm-atklabel_flip` | 5 | 0.081±0.060 | 15 | 0.110±0.056 | +0.028 | |

| `t1b-pipeline_krum-ssm-atksign_flip` | 5 | 0.050±0.049 | 15 | 0.076±0.048 | +0.027 | |

### bloco `t2a`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `t2a-fedavg-cnn_lstm-a0p1-atkadaptive_cosine` | 3 | 0.489±0.006 | 9 | 0.532±0.106 | +0.043 | |

| `t2a-fedavg-cnn_lstm-a0p1-atkclean` | 3 | 0.549±0.091 | 9 | 0.499±0.125 | -0.051 | ⚑ |

| `t2a-fedavg-cnn_lstm-a1p0-atkadaptive_cosine` | 3 | 0.655±0.049 | 9 | 0.630±0.079 | -0.025 | |

| `t2a-fedavg-cnn_lstm-a1p0-atkclean` | 3 | 0.666±0.033 | 9 | 0.649±0.041 | -0.016 | |

| `t2a-fedavg-cnn_lstm-aiid-atkadaptive_cosine` | 3 | 0.655±0.019 | 9 | 0.629±0.056 | -0.026 | |

| `t2a-fedavg-cnn_lstm-aiid-atkclean` | 3 | 0.665±0.033 | 9 | 0.683±0.030 | +0.018 | |

| `t2a-fedavg-snn-a0p1-atkadaptive_cosine` | 3 | 0.122±0.016 | 9 | 0.123±0.011 | +0.001 | |

| `t2a-fedavg-snn-a0p1-atkclean` | 3 | 0.124±0.019 | 9 | 0.118±0.009 | -0.006 | |

| `t2a-fedavg-snn-a1p0-atkadaptive_cosine` | 3 | 0.169±0.010 | 9 | 0.171±0.022 | +0.001 | |

| `t2a-fedavg-snn-a1p0-atkclean` | 3 | 0.174±0.005 | 9 | 0.170±0.015 | -0.003 | |

| `t2a-fedavg-snn-aiid-atkadaptive_cosine` | 3 | 0.215±0.026 | 9 | 0.199±0.026 | -0.016 | |

| `t2a-fedavg-snn-aiid-atkclean` | 3 | 0.204±0.005 | 9 | 0.190±0.022 | -0.015 | |

| `t2a-fedavg-ssm-a0p1-atkadaptive_cosine` | 3 | 0.678±0.062 | 9 | 0.687±0.069 | +0.009 | |

| `t2a-fedavg-ssm-a0p1-atkclean` | 3 | 0.774±0.082 | 9 | 0.798±0.059 | +0.024 | |

| `t2a-fedavg-ssm-a1p0-atkadaptive_cosine` | 3 | 0.745±0.022 | 9 | 0.744±0.028 | -0.001 | |

| `t2a-fedavg-ssm-a1p0-atkclean` | 3 | 0.748±0.022 | 9 | 0.757±0.020 | +0.008 | |

| `t2a-fedavg-ssm-aiid-atkadaptive_cosine` | 3 | 0.771±0.024 | 9 | 0.763±0.024 | -0.008 | |

| `t2a-fedavg-ssm-aiid-atkclean` | 3 | 0.756±0.040 | 9 | 0.770±0.027 | +0.014 | |

| `t2a-pipeline_krum-cnn_lstm-a0p1-atkadaptive_cosine` | 3 | 0.168±0.114 | 9 | 0.171±0.051 | +0.004 | |

| `t2a-pipeline_krum-cnn_lstm-a0p1-atkclean` | 3 | 0.347±0.125 | 9 | 0.300±0.097 | -0.047 | |

| `t2a-pipeline_krum-cnn_lstm-a1p0-atkadaptive_cosine` | 3 | 0.230±0.175 | 9 | 0.233±0.129 | +0.003 | |

| `t2a-pipeline_krum-cnn_lstm-a1p0-atkclean` | 3 | 0.468±0.056 | 9 | 0.477±0.048 | +0.009 | |

| `t2a-pipeline_krum-cnn_lstm-aiid-atkadaptive_cosine` | 3 | 0.260±0.076 | 9 | 0.282±0.103 | +0.023 | |

| `t2a-pipeline_krum-cnn_lstm-aiid-atkclean` | 3 | 0.548±0.050 | 9 | 0.556±0.073 | +0.009 | |

| `t2a-pipeline_krum-snn-a0p1-atkadaptive_cosine` | 3 | 0.127±0.010 | 9 | 0.091±0.028 | -0.036 | |

| `t2a-pipeline_krum-snn-a0p1-atkclean` | 3 | 0.119±0.007 | 9 | 0.113±0.011 | -0.007 | |

| `t2a-pipeline_krum-snn-a1p0-atkadaptive_cosine` | 3 | 0.100±0.030 | 9 | 0.100±0.022 | -0.001 | |

| `t2a-pipeline_krum-snn-a1p0-atkclean` | 3 | 0.133±0.007 | 9 | 0.131±0.014 | -0.002 | |

| `t2a-pipeline_krum-snn-aiid-atkadaptive_cosine` | 3 | 0.145±0.019 | 9 | 0.109±0.040 | -0.037 | |

| `t2a-pipeline_krum-snn-aiid-atkclean` | 3 | 0.152±0.025 | 9 | 0.139±0.014 | -0.013 | |

| `t2a-pipeline_krum-ssm-a0p1-atkadaptive_cosine` | 3 | 0.063±0.023 | 9 | 0.051±0.041 | -0.012 | |

| `t2a-pipeline_krum-ssm-a0p1-atkclean` | 3 | 0.272±0.122 | 9 | 0.206±0.087 | -0.066 | ⚑ |

| `t2a-pipeline_krum-ssm-a1p0-atkadaptive_cosine` | 3 | 0.015±0.007 | 9 | 0.030±0.029 | +0.014 | |

| `t2a-pipeline_krum-ssm-a1p0-atkclean` | 3 | 0.251±0.091 | 9 | 0.180±0.087 | -0.071 | ⚑ |

| `t2a-pipeline_krum-ssm-aiid-atkadaptive_cosine` | 3 | 0.049±0.026 | 9 | 0.050±0.031 | +0.001 | |

| `t2a-pipeline_krum-ssm-aiid-atkclean` | 3 | 0.237±0.118 | 9 | 0.279±0.130 | +0.041 | |

### bloco `t2b`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `t2b-no_defense-cnn_lstm-atkadaptive_cosine-frac1` | 3 | 0.581±0.028 | 9 | 0.616±0.059 | +0.035 | |

| `t2b-no_defense-cnn_lstm-atkadaptive_cosine-frac2` | 3 | 0.546±0.088 | 9 | 0.614±0.067 | +0.068 | ⚑ |

| `t2b-no_defense-cnn_lstm-atksign_flip-frac1` | 3 | 0.500±0.057 | 9 | 0.554±0.075 | +0.055 | ⚑ |

| `t2b-no_defense-cnn_lstm-atksign_flip-frac2` | 3 | 0.420±0.025 | 9 | 0.475±0.080 | +0.055 | ⚑ |

| `t2b-no_defense-snn-atkadaptive_cosine-frac1` | 3 | 0.154±0.001 | 9 | 0.155±0.012 | +0.002 | |

| `t2b-no_defense-snn-atkadaptive_cosine-frac2` | 3 | 0.143±0.007 | 9 | 0.150±0.017 | +0.007 | |

| `t2b-no_defense-snn-atksign_flip-frac1` | 3 | 0.139±0.012 | 9 | 0.144±0.017 | +0.005 | |

| `t2b-no_defense-snn-atksign_flip-frac2` | 3 | 0.134±0.019 | 9 | 0.131±0.011 | -0.003 | |

| `t2b-no_defense-ssm-atkadaptive_cosine-frac1` | 3 | 0.723±0.061 | 9 | 0.758±0.049 | +0.035 | |

| `t2b-no_defense-ssm-atkadaptive_cosine-frac2` | 3 | 0.686±0.066 | 9 | 0.733±0.060 | +0.047 | |

| `t2b-no_defense-ssm-atksign_flip-frac1` | 3 | 0.540±0.091 | 9 | 0.559±0.066 | +0.019 | |

| `t2b-no_defense-ssm-atksign_flip-frac2` | 3 | 0.398±0.089 | 9 | 0.441±0.080 | +0.043 | |

| `t2b-pipeline_krum-cnn_lstm-atkadaptive_cosine-frac1` | 3 | 0.462±0.060 | 9 | 0.433±0.041 | -0.029 | |

| `t2b-pipeline_krum-cnn_lstm-atkadaptive_cosine-frac2` | 3 | 0.418±0.073 | 9 | 0.365±0.092 | -0.053 | ⚑ |

| `t2b-pipeline_krum-cnn_lstm-atksign_flip-frac1` | 3 | 0.416±0.068 | 9 | 0.493±0.037 | +0.077 | ⚑ |

| `t2b-pipeline_krum-cnn_lstm-atksign_flip-frac2` | 3 | 0.349±0.036 | 9 | 0.358±0.092 | +0.009 | |

| `t2b-pipeline_krum-snn-atkadaptive_cosine-frac1` | 3 | 0.147±0.026 | 9 | 0.127±0.025 | -0.020 | |

| `t2b-pipeline_krum-snn-atkadaptive_cosine-frac2` | 3 | 0.127±0.026 | 9 | 0.110±0.017 | -0.017 | |

| `t2b-pipeline_krum-snn-atksign_flip-frac1` | 3 | 0.116±0.007 | 9 | 0.118±0.010 | +0.002 | |

| `t2b-pipeline_krum-snn-atksign_flip-frac2` | 3 | 0.123±0.027 | 9 | 0.110±0.018 | -0.014 | |

| `t2b-pipeline_krum-ssm-atkadaptive_cosine-frac1` | 3 | 0.161±0.027 | 9 | 0.165±0.034 | +0.005 | |

| `t2b-pipeline_krum-ssm-atkadaptive_cosine-frac2` | 3 | 0.219±0.075 | 9 | 0.207±0.065 | -0.011 | |

| `t2b-pipeline_krum-ssm-atksign_flip-frac1` | 3 | 0.201±0.069 | 9 | 0.173±0.056 | -0.028 | |

| `t2b-pipeline_krum-ssm-atksign_flip-frac2` | 3 | 0.153±0.070 | 9 | 0.140±0.042 | -0.012 | |

### bloco `t2c`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `t2c-pipeline_krum-cnn_lstm-sg0p25-t0p3` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg0p25-t0p5` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg0p25-t0p7` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg0p5-t0p3` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg0p5-t0p5` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg0p5-t0p7` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg1p0-t0p3` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg1p0-t0p5` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg1p0-t0p7` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg2p0-t0p3` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg2p0-t0p5` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-cnn_lstm-sg2p0-t0p7` | 3 | 0.054±0.061 | 9 | 0.024±0.038 | -0.030 | |

| `t2c-pipeline_krum-snn-sg0p25-t0p3` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg0p25-t0p5` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg0p25-t0p7` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg0p5-t0p3` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg0p5-t0p5` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg0p5-t0p7` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg1p0-t0p3` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg1p0-t0p5` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg1p0-t0p7` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg2p0-t0p3` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg2p0-t0p5` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-snn-sg2p0-t0p7` | 3 | 0.009±0.007 | 9 | 0.040±0.042 | +0.030 | |

| `t2c-pipeline_krum-ssm-sg0p25-t0p3` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg0p25-t0p5` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg0p25-t0p7` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg0p5-t0p3` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg0p5-t0p5` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg0p5-t0p7` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg1p0-t0p3` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg1p0-t0p5` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg1p0-t0p7` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg2p0-t0p3` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg2p0-t0p5` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t2c-pipeline_krum-ssm-sg2p0-t0p7` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

### bloco `t2d`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `t2d-pipeline_krum-cnn_lstm-m4` | 3 | 0.334±0.009 | 9 | 0.210±0.121 | -0.124 | ⚑ |

| `t2d-pipeline_krum-cnn_lstm-m7` | 3 | 0.304±0.051 | 9 | 0.220±0.097 | -0.084 | ⚑ |

| `t2d-pipeline_krum-snn-m4` | 3 | 0.100±0.014 | 9 | 0.103±0.036 | +0.003 | |

| `t2d-pipeline_krum-snn-m7` | 3 | 0.121±0.006 | 9 | 0.106±0.032 | -0.015 | |

| `t2d-pipeline_krum-ssm-m4` | 3 | 0.037±0.031 | 9 | 0.023±0.018 | -0.014 | |

| `t2d-pipeline_krum-ssm-m7` | 3 | 0.036±0.032 | 9 | 0.024±0.020 | -0.012 | |

### bloco `t2e`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `t2e-fedavg-snn-codingrate-atkadaptive_cosine` | 3 | 0.120±0.003 | 9 | 0.121±0.001 | +0.001 | |

| `t2e-fedavg-snn-codingrate-atkclean` | 3 | 0.121±0.001 | 9 | 0.121±0.000 | -0.000 | |

| `t2e-fedavg-snn-codingrate-ts16-atkadaptive_cosine` | 3 | 0.123±0.010 | 0 | — | — | |

| `t2e-fedavg-snn-codingrate-ts16-atkclean` | 3 | 0.121±0.001 | 0 | — | — | |

| `t2e-fedavg-snn-codingrate-ts4-atkadaptive_cosine` | 3 | 0.122±0.001 | 0 | — | — | |

| `t2e-fedavg-snn-codingrate-ts4-atkclean` | 3 | 0.125±0.007 | 0 | — | — | |

| `t2e-fedavg-snn-ts16-atkadaptive_cosine` | 3 | 0.128±0.008 | 9 | 0.121±0.008 | -0.007 | |

| `t2e-fedavg-snn-ts16-atkclean` | 3 | 0.134±0.003 | 9 | 0.132±0.016 | -0.002 | |

| `t2e-fedavg-snn-ts4-atkadaptive_cosine` | 3 | 0.188±0.025 | 9 | 0.192±0.020 | +0.005 | |

| `t2e-fedavg-snn-ts4-atkclean` | 3 | 0.196±0.022 | 9 | 0.204±0.037 | +0.008 | |

| `t2e-fedavg-ssm-codingrate-atkadaptive_cosine` | 3 | 0.677±0.064 | 9 | 0.726±0.055 | +0.049 | |

| `t2e-fedavg-ssm-codingrate-atkclean` | 3 | 0.733±0.042 | 9 | 0.764±0.040 | +0.031 | |

| `t2e-fedavg-ssm-ts16-atkadaptive_cosine` | 3 | 0.677±0.062 | 0 | — | — | |

| `t2e-fedavg-ssm-ts16-atkclean` | 3 | 0.733±0.041 | 0 | — | — | |

| `t2e-fedavg-ssm-ts4-atkadaptive_cosine` | 3 | 0.677±0.062 | 0 | — | — | |

| `t2e-fedavg-ssm-ts4-atkclean` | 3 | 0.733±0.041 | 0 | — | — | |

| `t2e-fedprox-cnn_lstm-mu0p001` | 3 | 0.571±0.066 | 9 | 0.596±0.064 | +0.024 | |

| `t2e-fedprox-cnn_lstm-mu0p1` | 3 | 0.546±0.006 | 9 | 0.586±0.052 | +0.040 | |

| `t2e-fedprox-snn-mu0p001` | 3 | 0.154±0.009 | 9 | 0.157±0.022 | +0.003 | |

| `t2e-fedprox-snn-mu0p1` | 3 | 0.161±0.012 | 9 | 0.164±0.019 | +0.002 | |

| `t2e-fedprox-ssm-mu0p001` | 3 | 0.731±0.050 | 9 | 0.761±0.039 | +0.030 | |

| `t2e-fedprox-ssm-mu0p1` | 3 | 0.587±0.053 | 9 | 0.614±0.052 | +0.027 | |

| `t2e-fidelity_only-cnn_lstm-atkadaptive_cosine` | 3 | 0.474±0.025 | 9 | 0.396±0.110 | -0.078 | ⚑ |

| `t2e-fidelity_only-snn-atkadaptive_cosine` | 3 | 0.044±0.065 | 9 | 0.106±0.034 | +0.062 | ⚑ |

| `t2e-fidelity_only-ssm-atkadaptive_cosine` | 3 | 0.109±0.036 | 9 | 0.064±0.038 | -0.045 | |

### bloco `t3tau`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `t3tau-tau0p6-t1b-pipeline_geomed-cnn_lstm-atkadaptive_cosine` | 3 | 0.237±0.263 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_geomed-cnn_lstm-atkgaussian_noise` | 3 | 0.505±0.031 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_geomed-cnn_lstm-atklabel_flip` | 3 | 0.404±0.153 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_geomed-cnn_lstm-atksign_flip` | 3 | 0.397±0.039 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_geomed-snn-atkadaptive_cosine` | 3 | 0.043±0.063 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_geomed-snn-atkgaussian_noise` | 3 | 0.148±0.003 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_geomed-snn-atklabel_flip` | 3 | 0.127±0.016 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_geomed-snn-atksign_flip` | 3 | 0.130±0.010 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_geomed-ssm-atkadaptive_cosine` | 3 | 0.058±0.034 | 9 | 0.031±0.017 | -0.028 | |

| `t3tau-tau0p6-t1b-pipeline_geomed-ssm-atkgaussian_noise` | 3 | 0.243±0.088 | 9 | 0.180±0.081 | -0.063 | ⚑ |

| `t3tau-tau0p6-t1b-pipeline_geomed-ssm-atklabel_flip` | 3 | 0.198±0.039 | 9 | 0.211±0.094 | +0.014 | |

| `t3tau-tau0p6-t1b-pipeline_geomed-ssm-atksign_flip` | 3 | 0.183±0.061 | 9 | 0.154±0.045 | -0.029 | |

| `t3tau-tau0p6-t1b-pipeline_krum-cnn_lstm-atkadaptive_cosine` | 3 | 0.191±0.079 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_krum-cnn_lstm-atkgaussian_noise` | 3 | 0.408±0.065 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_krum-cnn_lstm-atklabel_flip` | 3 | 0.435±0.109 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_krum-cnn_lstm-atksign_flip` | 3 | 0.472±0.117 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_krum-snn-atkadaptive_cosine` | 3 | 0.072±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_krum-snn-atkgaussian_noise` | 3 | 0.110±0.004 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_krum-snn-atklabel_flip` | 3 | 0.115±0.021 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_krum-snn-atksign_flip` | 3 | 0.114±0.011 | 0 | — | — | |

| `t3tau-tau0p6-t1b-pipeline_krum-ssm-atkadaptive_cosine` | 3 | 0.064±0.049 | 9 | 0.054±0.042 | -0.009 | |

| `t3tau-tau0p6-t1b-pipeline_krum-ssm-atkgaussian_noise` | 3 | 0.269±0.016 | 9 | 0.286±0.055 | +0.017 | |

| `t3tau-tau0p6-t1b-pipeline_krum-ssm-atklabel_flip` | 3 | 0.266±0.143 | 9 | 0.291±0.097 | +0.026 | |

| `t3tau-tau0p6-t1b-pipeline_krum-ssm-atksign_flip` | 3 | 0.221±0.050 | 9 | 0.203±0.040 | -0.018 | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg0p25-t0p3` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg0p25-t0p5` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg0p25-t0p7` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg0p5-t0p3` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg0p5-t0p5` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg0p5-t0p7` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg1p0-t0p3` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg1p0-t0p5` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg1p0-t0p7` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg2p0-t0p3` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg2p0-t0p5` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-cnn_lstm-sg2p0-t0p7` | 3 | 0.054±0.061 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg0p25-t0p3` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg0p25-t0p5` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg0p25-t0p7` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg0p5-t0p3` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg0p5-t0p5` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg0p5-t0p7` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg1p0-t0p3` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg1p0-t0p5` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg1p0-t0p7` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg2p0-t0p3` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg2p0-t0p5` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-snn-sg2p0-t0p7` | 3 | 0.009±0.007 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg0p25-t0p3` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg0p25-t0p5` | 3 | 0.033±0.040 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg0p25-t0p7` | 3 | 0.033±0.040 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg0p5-t0p3` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg0p5-t0p5` | 3 | 0.033±0.040 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg0p5-t0p7` | 3 | 0.033±0.040 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg1p0-t0p3` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg1p0-t0p5` | 3 | 0.033±0.040 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg1p0-t0p7` | 3 | 0.033±0.040 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg2p0-t0p3` | 3 | 0.033±0.040 | 9 | 0.033±0.029 | -0.000 | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg2p0-t0p5` | 3 | 0.033±0.040 | 0 | — | — | |

| `t3tau-tau0p6-t2c-pipeline_krum-ssm-sg2p0-t0p7` | 3 | 0.033±0.040 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_geomed-cnn_lstm-atkadaptive_cosine` | 3 | 0.544±0.049 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_geomed-cnn_lstm-atkgaussian_noise` | 3 | 0.509±0.008 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_geomed-cnn_lstm-atklabel_flip` | 3 | 0.443±0.043 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_geomed-cnn_lstm-atksign_flip` | 3 | 0.443±0.079 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_geomed-snn-atkadaptive_cosine` | 3 | 0.143±0.011 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_geomed-snn-atkgaussian_noise` | 3 | 0.141±0.011 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_geomed-snn-atklabel_flip` | 3 | 0.068±0.063 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_geomed-snn-atksign_flip` | 3 | 0.138±0.015 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_geomed-ssm-atkadaptive_cosine` | 3 | 0.624±0.049 | 9 | 0.643±0.082 | +0.019 | |

| `t3tau-tau0p8-t1b-pipeline_geomed-ssm-atkgaussian_noise` | 3 | 0.623±0.092 | 9 | 0.684±0.068 | +0.061 | ⚑ |

| `t3tau-tau0p8-t1b-pipeline_geomed-ssm-atklabel_flip` | 3 | 0.484±0.153 | 9 | 0.480±0.111 | -0.005 | |

| `t3tau-tau0p8-t1b-pipeline_geomed-ssm-atksign_flip` | 3 | 0.391±0.097 | 9 | 0.440±0.069 | +0.049 | |

| `t3tau-tau0p8-t1b-pipeline_krum-cnn_lstm-atkadaptive_cosine` | 3 | 0.039±0.033 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_krum-cnn_lstm-atkgaussian_noise` | 3 | 0.397±0.119 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_krum-cnn_lstm-atklabel_flip` | 3 | 0.380±0.068 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_krum-cnn_lstm-atksign_flip` | 3 | 0.369±0.134 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_krum-snn-atkadaptive_cosine` | 3 | 0.045±0.067 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_krum-snn-atkgaussian_noise` | 3 | 0.136±0.006 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_krum-snn-atklabel_flip` | 3 | 0.125±0.044 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_krum-snn-atksign_flip` | 3 | 0.123±0.003 | 0 | — | — | |

| `t3tau-tau0p8-t1b-pipeline_krum-ssm-atkadaptive_cosine` | 3 | 0.060±0.037 | 9 | 0.056±0.035 | -0.004 | |

| `t3tau-tau0p8-t1b-pipeline_krum-ssm-atkgaussian_noise` | 3 | 0.270±0.062 | 9 | 0.301±0.053 | +0.030 | |

| `t3tau-tau0p8-t1b-pipeline_krum-ssm-atklabel_flip` | 3 | 0.321±0.056 | 9 | 0.336±0.044 | +0.015 | |

| `t3tau-tau0p8-t1b-pipeline_krum-ssm-atksign_flip` | 3 | 0.265±0.058 | 9 | 0.297±0.069 | +0.032 | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg0p25-t0p3` | 3 | 0.040±0.031 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg0p25-t0p5` | 3 | 0.039±0.030 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg0p25-t0p7` | 3 | 0.039±0.031 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg0p5-t0p3` | 3 | 0.036±0.034 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg0p5-t0p5` | 3 | 0.036±0.035 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg0p5-t0p7` | 3 | 0.036±0.034 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg1p0-t0p3` | 3 | 0.038±0.031 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg1p0-t0p5` | 3 | 0.038±0.031 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg1p0-t0p7` | 3 | 0.038±0.031 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg2p0-t0p3` | 3 | 0.038±0.033 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg2p0-t0p5` | 3 | 0.038±0.033 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-cnn_lstm-sg2p0-t0p7` | 3 | 0.038±0.033 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg0p25-t0p3` | 3 | 0.043±0.064 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg0p25-t0p5` | 3 | 0.043±0.064 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg0p25-t0p7` | 3 | 0.043±0.064 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg0p5-t0p3` | 3 | 0.043±0.065 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg0p5-t0p5` | 3 | 0.043±0.065 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg0p5-t0p7` | 3 | 0.043±0.065 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg1p0-t0p3` | 3 | 0.043±0.063 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg1p0-t0p5` | 3 | 0.043±0.063 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg1p0-t0p7` | 3 | 0.043±0.063 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg2p0-t0p3` | 3 | 0.042±0.061 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg2p0-t0p5` | 3 | 0.042±0.061 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-snn-sg2p0-t0p7` | 3 | 0.042±0.061 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg0p25-t0p3` | 3 | 0.082±0.068 | 9 | 0.016±0.040 | -0.066 | ⚑ |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg0p25-t0p5` | 3 | 0.082±0.068 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg0p25-t0p7` | 3 | 0.082±0.068 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg0p5-t0p3` | 3 | 0.082±0.068 | 9 | 0.016±0.040 | -0.066 | ⚑ |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg0p5-t0p5` | 3 | 0.082±0.068 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg0p5-t0p7` | 3 | 0.082±0.068 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg1p0-t0p3` | 3 | 0.080±0.070 | 9 | 0.068±0.054 | -0.012 | ·div1→2 |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg1p0-t0p5` | 3 | 0.080±0.070 | 0 | — | — | ·div1→0 |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg1p0-t0p7` | 3 | 0.080±0.070 | 0 | — | — | ·div1→0 |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg2p0-t0p3` | 3 | 0.121±0.001 | 9 | 0.084±0.051 | -0.037 | ·div0→1 |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg2p0-t0p5` | 3 | 0.121±0.001 | 0 | — | — | |

| `t3tau-tau0p8-t2c-pipeline_krum-ssm-sg2p0-t0p7` | 3 | 0.121±0.001 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_geomed-cnn_lstm-atkadaptive_cosine` | 3 | 0.581±0.055 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_geomed-cnn_lstm-atkgaussian_noise` | 3 | 0.577±0.041 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_geomed-cnn_lstm-atklabel_flip` | 3 | 0.453±0.006 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_geomed-cnn_lstm-atksign_flip` | 3 | 0.402±0.044 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_geomed-snn-atkadaptive_cosine` | 3 | 0.144±0.008 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_geomed-snn-atkgaussian_noise` | 3 | 0.142±0.013 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_geomed-snn-atklabel_flip` | 3 | 0.089±0.022 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_geomed-snn-atksign_flip` | 3 | 0.131±0.006 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_geomed-ssm-atkadaptive_cosine` | 3 | 0.689±0.056 | 9 | 0.718±0.042 | +0.029 | |

| `t3tau-tau1p0-t1b-pipeline_geomed-ssm-atkgaussian_noise` | 3 | 0.626±0.089 | 9 | 0.682±0.067 | +0.056 | ⚑ |

| `t3tau-tau1p0-t1b-pipeline_geomed-ssm-atklabel_flip` | 3 | 0.410±0.007 | 9 | 0.436±0.047 | +0.026 | |

| `t3tau-tau1p0-t1b-pipeline_geomed-ssm-atksign_flip` | 3 | 0.336±0.079 | 9 | 0.384±0.074 | +0.048 | |

| `t3tau-tau1p0-t1b-pipeline_krum-cnn_lstm-atkadaptive_cosine` | 3 | 0.070±0.039 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_krum-cnn_lstm-atkgaussian_noise` | 3 | 0.357±0.125 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_krum-cnn_lstm-atklabel_flip` | 3 | 0.343±0.156 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_krum-cnn_lstm-atksign_flip` | 3 | 0.376±0.112 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_krum-snn-atkadaptive_cosine` | 3 | 0.117±0.007 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_krum-snn-atkgaussian_noise` | 3 | 0.137±0.002 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_krum-snn-atklabel_flip` | 3 | 0.129±0.030 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_krum-snn-atksign_flip` | 3 | 0.132±0.014 | 0 | — | — | |

| `t3tau-tau1p0-t1b-pipeline_krum-ssm-atkadaptive_cosine` | 3 | 0.060±0.037 | 9 | 0.056±0.035 | -0.004 | |

| `t3tau-tau1p0-t1b-pipeline_krum-ssm-atkgaussian_noise` | 3 | 0.271±0.057 | 9 | 0.280±0.045 | +0.009 | |

| `t3tau-tau1p0-t1b-pipeline_krum-ssm-atklabel_flip` | 3 | 0.293±0.067 | 9 | 0.292±0.054 | -0.000 | |

| `t3tau-tau1p0-t1b-pipeline_krum-ssm-atksign_flip` | 3 | 0.215±0.096 | 9 | 0.228±0.049 | +0.014 | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg0p25-t0p3` | 3 | 0.039±0.030 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg0p25-t0p5` | 3 | 0.040±0.030 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg0p25-t0p7` | 3 | 0.039±0.030 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg0p5-t0p3` | 3 | 0.036±0.034 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg0p5-t0p5` | 3 | 0.036±0.034 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg0p5-t0p7` | 3 | 0.036±0.034 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg1p0-t0p3` | 3 | 0.038±0.031 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg1p0-t0p5` | 3 | 0.038±0.031 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg1p0-t0p7` | 3 | 0.038±0.031 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg2p0-t0p3` | 3 | 0.038±0.033 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg2p0-t0p5` | 3 | 0.038±0.033 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-cnn_lstm-sg2p0-t0p7` | 3 | 0.038±0.033 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg0p25-t0p3` | 3 | 0.043±0.064 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg0p25-t0p5` | 3 | 0.043±0.064 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg0p25-t0p7` | 3 | 0.043±0.064 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg0p5-t0p3` | 3 | 0.043±0.064 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg0p5-t0p5` | 3 | 0.043±0.064 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg0p5-t0p7` | 3 | 0.043±0.064 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg1p0-t0p3` | 3 | 0.043±0.063 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg1p0-t0p5` | 3 | 0.043±0.063 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg1p0-t0p7` | 3 | 0.043±0.063 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg2p0-t0p3` | 3 | 0.042±0.061 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg2p0-t0p5` | 3 | 0.042±0.061 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-snn-sg2p0-t0p7` | 3 | 0.042±0.061 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg0p25-t0p3` | 3 | 0.082±0.068 | 9 | 0.016±0.040 | -0.066 | ⚑ |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg0p25-t0p5` | 3 | 0.082±0.068 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg0p25-t0p7` | 3 | 0.082±0.068 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg0p5-t0p3` | 3 | 0.082±0.068 | 9 | 0.016±0.040 | -0.066 | ⚑ |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg0p5-t0p5` | 3 | 0.082±0.068 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg0p5-t0p7` | 3 | 0.082±0.068 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg1p0-t0p3` | 3 | 0.080±0.070 | 9 | 0.094±0.053 | +0.013 | ·div1→2 |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg1p0-t0p5` | 3 | 0.080±0.070 | 0 | — | — | ·div1→0 |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg1p0-t0p7` | 3 | 0.080±0.070 | 0 | — | — | ·div1→0 |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg2p0-t0p3` | 3 | 0.121±0.001 | 9 | 0.107±0.040 | -0.014 | ·div0→1 |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg2p0-t0p5` | 3 | 0.121±0.001 | 0 | — | — | |

| `t3tau-tau1p0-t2c-pipeline_krum-ssm-sg2p0-t0p7` | 3 | 0.121±0.001 | 0 | — | — | |

### bloco `t4mag`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `t4mag-geomed_alone-cnn_lstm-sc10p0` | 3 | 0.606±0.079 | 0 | — | — | |

| `t4mag-geomed_alone-cnn_lstm-sc1p0` | 3 | 0.620±0.081 | 0 | — | — | |

| `t4mag-geomed_alone-cnn_lstm-sc2p0` | 3 | 0.593±0.013 | 0 | — | — | |

| `t4mag-geomed_alone-cnn_lstm-sc5p0` | 3 | 0.599±0.077 | 0 | — | — | |

| `t4mag-geomed_alone-snn-sc10p0` | 3 | 0.144±0.009 | 0 | — | — | |

| `t4mag-geomed_alone-snn-sc1p0` | 3 | 0.144±0.008 | 0 | — | — | |

| `t4mag-geomed_alone-snn-sc2p0` | 3 | 0.148±0.012 | 0 | — | — | |

| `t4mag-geomed_alone-snn-sc5p0` | 3 | 0.147±0.015 | 0 | — | — | |

| `t4mag-geomed_alone-ssm-sc10p0` | 3 | 0.711±0.031 | 0 | — | — | |

| `t4mag-geomed_alone-ssm-sc1p0` | 3 | 0.693±0.059 | 0 | — | — | |

| `t4mag-geomed_alone-ssm-sc2p0` | 3 | 0.685±0.060 | 0 | — | — | |

| `t4mag-geomed_alone-ssm-sc5p0` | 3 | 0.701±0.067 | 0 | — | — | |

| `t4mag-no_defense-cnn_lstm-sc10p0` | 3 | 0.103±0.030 | 0 | — | — | |

| `t4mag-no_defense-cnn_lstm-sc1p0` | 3 | 0.628±0.043 | 0 | — | — | |

| `t4mag-no_defense-cnn_lstm-sc2p0` | 3 | 0.529±0.118 | 0 | — | — | |

| `t4mag-no_defense-cnn_lstm-sc5p0` | 3 | 0.118±0.020 | 0 | — | — | |

| `t4mag-no_defense-snn-sc10p0` | 3 | 0.115±0.009 | 0 | — | — | |

| `t4mag-no_defense-snn-sc1p0` | 3 | 0.142±0.008 | 0 | — | — | |

| `t4mag-no_defense-snn-sc2p0` | 3 | 0.144±0.010 | 0 | — | — | |

| `t4mag-no_defense-snn-sc5p0` | 3 | 0.126±0.005 | 0 | — | — | |

| `t4mag-no_defense-ssm-sc10p0` | 3 | 0.189±0.038 | 0 | — | — | |

| `t4mag-no_defense-ssm-sc1p0` | 3 | 0.679±0.062 | 0 | — | — | |

| `t4mag-no_defense-ssm-sc2p0` | 3 | 0.702±0.067 | 0 | — | — | |

| `t4mag-no_defense-ssm-sc5p0` | 3 | 0.228±0.046 | 0 | — | — | |

| `t4mag-pipeline_geomed-cnn_lstm-sc10p0` | 3 | 0.081±0.008 | 0 | — | — | |

| `t4mag-pipeline_geomed-cnn_lstm-sc1p0` | 3 | 0.318±0.151 | 0 | — | — | |

| `t4mag-pipeline_geomed-cnn_lstm-sc2p0` | 3 | 0.313±0.084 | 0 | — | — | |

| `t4mag-pipeline_geomed-cnn_lstm-sc5p0` | 3 | 0.202±0.085 | 0 | — | — | |

| `t4mag-pipeline_geomed-snn-sc10p0` | 3 | 0.108±0.023 | 0 | — | — | |

| `t4mag-pipeline_geomed-snn-sc1p0` | 3 | 0.099±0.038 | 0 | — | — | |

| `t4mag-pipeline_geomed-snn-sc2p0` | 3 | 0.117±0.004 | 0 | — | — | |

| `t4mag-pipeline_geomed-snn-sc5p0` | 3 | 0.117±0.003 | 0 | — | — | |

| `t4mag-pipeline_geomed-ssm-sc10p0` | 3 | 0.032±0.019 | 0 | — | — | |

| `t4mag-pipeline_geomed-ssm-sc1p0` | 3 | 0.071±0.030 | 0 | — | — | |

| `t4mag-pipeline_geomed-ssm-sc2p0` | 3 | 0.059±0.056 | 0 | — | — | |

| `t4mag-pipeline_geomed-ssm-sc5p0` | 3 | 0.047±0.031 | 0 | — | — | |

| `t4mag-pipeline_krum-cnn_lstm-sc10p0` | 3 | 0.059±0.038 | 0 | — | — | |

| `t4mag-pipeline_krum-cnn_lstm-sc1p0` | 3 | 0.224±0.001 | 0 | — | — | |

| `t4mag-pipeline_krum-cnn_lstm-sc2p0` | 3 | 0.149±0.094 | 0 | — | — | |

| `t4mag-pipeline_krum-cnn_lstm-sc5p0` | 3 | 0.017±0.006 | 0 | — | — | |

| `t4mag-pipeline_krum-snn-sc10p0` | 3 | 0.072±0.062 | 0 | — | — | |

| `t4mag-pipeline_krum-snn-sc1p0` | 3 | 0.115±0.012 | 0 | — | — | |

| `t4mag-pipeline_krum-snn-sc2p0` | 3 | 0.081±0.064 | 0 | — | — | |

| `t4mag-pipeline_krum-snn-sc5p0` | 3 | 0.079±0.067 | 0 | — | — | |

| `t4mag-pipeline_krum-ssm-sc10p0` | 3 | 0.045±0.054 | 0 | — | — | |

| `t4mag-pipeline_krum-ssm-sc1p0` | 3 | 0.037±0.030 | 0 | — | — | |

| `t4mag-pipeline_krum-ssm-sc2p0` | 3 | 0.035±0.045 | 0 | — | — | |

| `t4mag-pipeline_krum-ssm-sc5p0` | 3 | 0.037±0.039 | 0 | — | — | |

### bloco `temporal-t1a`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `temporal-t1a-centralized-cnn_lstm` | 5 | 0.522±0.083 | 0 | — | — | |

| `temporal-t1a-centralized-snn` | 5 | 0.218±0.025 | 0 | — | — | |

| `temporal-t1a-centralized-ssm` | 5 | 0.728±0.051 | 0 | — | — | |

| `temporal-t1a-fedavg-cnn_lstm` | 5 | 0.557±0.095 | 0 | — | — | |

| `temporal-t1a-fedavg-snn` | 5 | 0.124±0.011 | 0 | — | — | |

| `temporal-t1a-fedavg-ssm` | 5 | 0.582±0.038 | 0 | — | — | |

| `temporal-t1a-fedprox-cnn_lstm` | 5 | 0.557±0.066 | 0 | — | — | |

| `temporal-t1a-fedprox-snn` | 5 | 0.129±0.018 | 0 | — | — | |

| `temporal-t1a-fedprox-ssm` | 5 | 0.584±0.046 | 0 | — | — | |

| `temporal-t1a-local-cnn_lstm` | 5 | 0.270±0.029 | 0 | — | — | |

| `temporal-t1a-local-snn` | 5 | 0.106±0.003 | 0 | — | — | |

| `temporal-t1a-local-ssm` | 5 | 0.159±0.011 | 0 | — | — | |

| `temporal-t1a-pipeline_geomed-cnn_lstm` | 5 | 0.399±0.081 | 0 | — | — | |

| `temporal-t1a-pipeline_geomed-snn` | 5 | 0.114±0.013 | 0 | — | — | |

| `temporal-t1a-pipeline_geomed-ssm` | 5 | 0.129±0.015 | 0 | — | — | |

| `temporal-t1a-pipeline_krum-cnn_lstm` | 5 | 0.343±0.066 | 0 | — | — | |

| `temporal-t1a-pipeline_krum-snn` | 5 | 0.110±0.012 | 0 | — | — | |

| `temporal-t1a-pipeline_krum-ssm` | 5 | 0.132±0.032 | 0 | — | — | |

### bloco `temporal-t1b`

| template | n_old | old mean±sd | n_new | new mean±sd | Δ | |
|---|---|---|---|---|---|---|
| `temporal-t1b-geomed_alone-cnn_lstm-atkadaptive_cosine` | 5 | 0.540±0.056 | 0 | — | — | |

| `temporal-t1b-geomed_alone-cnn_lstm-atkgaussian_noise` | 5 | 0.497±0.083 | 0 | — | — | |

| `temporal-t1b-geomed_alone-cnn_lstm-atklabel_flip` | 5 | 0.489±0.073 | 0 | — | — | |

| `temporal-t1b-geomed_alone-cnn_lstm-atksign_flip` | 5 | 0.361±0.085 | 0 | — | — | |

| `temporal-t1b-geomed_alone-snn-atkadaptive_cosine` | 5 | 0.125±0.006 | 0 | — | — | |

| `temporal-t1b-geomed_alone-snn-atkgaussian_noise` | 5 | 0.129±0.018 | 0 | — | — | |

| `temporal-t1b-geomed_alone-snn-atklabel_flip` | 5 | 0.074±0.032 | 0 | — | — | |

| `temporal-t1b-geomed_alone-snn-atksign_flip` | 5 | 0.108±0.006 | 0 | — | — | |

| `temporal-t1b-geomed_alone-ssm-atkadaptive_cosine` | 5 | 0.520±0.038 | 0 | — | — | |

| `temporal-t1b-geomed_alone-ssm-atkgaussian_noise` | 5 | 0.483±0.053 | 0 | — | — | |

| `temporal-t1b-geomed_alone-ssm-atklabel_flip` | 5 | 0.329±0.017 | 0 | — | — | |

| `temporal-t1b-geomed_alone-ssm-atksign_flip` | 5 | 0.222±0.051 | 0 | — | — | |

| `temporal-t1b-krum_alone-cnn_lstm-atkadaptive_cosine` | 5 | 0.130±0.088 | 0 | — | — | |

| `temporal-t1b-krum_alone-cnn_lstm-atkgaussian_noise` | 5 | 0.354±0.060 | 0 | — | — | |

| `temporal-t1b-krum_alone-cnn_lstm-atklabel_flip` | 5 | 0.354±0.060 | 0 | — | — | |

| `temporal-t1b-krum_alone-cnn_lstm-atksign_flip` | 5 | 0.344±0.048 | 0 | — | — | |

| `temporal-t1b-krum_alone-snn-atkadaptive_cosine` | 5 | 0.089±0.047 | 0 | — | — | |

| `temporal-t1b-krum_alone-snn-atkgaussian_noise` | 5 | 0.106±0.010 | 0 | — | — | |

| `temporal-t1b-krum_alone-snn-atklabel_flip` | 5 | 0.104±0.012 | 0 | — | — | |

| `temporal-t1b-krum_alone-snn-atksign_flip` | 5 | 0.109±0.022 | 0 | — | — | |

| `temporal-t1b-krum_alone-ssm-atkadaptive_cosine` | 5 | 0.045±0.024 | 0 | — | — | |

| `temporal-t1b-krum_alone-ssm-atkgaussian_noise` | 5 | 0.231±0.074 | 0 | — | — | |

| `temporal-t1b-krum_alone-ssm-atklabel_flip` | 5 | 0.222±0.093 | 0 | — | — | |

| `temporal-t1b-krum_alone-ssm-atksign_flip` | 5 | 0.180±0.064 | 0 | — | — | |

| `temporal-t1b-no_defense-cnn_lstm-atkadaptive_cosine` | 5 | 0.546±0.032 | 0 | — | — | |

| `temporal-t1b-no_defense-cnn_lstm-atkgaussian_noise` | 5 | 0.050±0.021 | 0 | — | — | |

| `temporal-t1b-no_defense-cnn_lstm-atklabel_flip` | 5 | 0.487±0.068 | 0 | — | — | |

| `temporal-t1b-no_defense-cnn_lstm-atksign_flip` | 5 | 0.293±0.083 | 0 | — | — | |

| `temporal-t1b-no_defense-snn-atkadaptive_cosine` | 5 | 0.122±0.015 | 0 | — | — | |

| `temporal-t1b-no_defense-snn-atkgaussian_noise` | 5 | 0.117±0.006 | 0 | — | — | |

| `temporal-t1b-no_defense-snn-atklabel_flip` | 5 | 0.074±0.029 | 0 | — | — | |

| `temporal-t1b-no_defense-snn-atksign_flip` | 5 | 0.106±0.012 | 0 | — | — | |

| `temporal-t1b-no_defense-ssm-atkadaptive_cosine` | 5 | 0.522±0.036 | 0 | — | — | |

| `temporal-t1b-no_defense-ssm-atkgaussian_noise` | 5 | 0.075±0.010 | 0 | — | — | |

| `temporal-t1b-no_defense-ssm-atklabel_flip` | 5 | 0.318±0.033 | 0 | — | — | |

| `temporal-t1b-no_defense-ssm-atksign_flip` | 5 | 0.174±0.050 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-cnn_lstm-atkadaptive_cosine` | 5 | 0.388±0.089 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-cnn_lstm-atkgaussian_noise` | 5 | 0.321±0.075 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-cnn_lstm-atklabel_flip` | 5 | 0.350±0.070 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-cnn_lstm-atksign_flip` | 5 | 0.273±0.072 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-snn-atkadaptive_cosine` | 5 | 0.097±0.044 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-snn-atkgaussian_noise` | 5 | 0.114±0.008 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-snn-atklabel_flip` | 5 | 0.096±0.018 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-snn-atksign_flip` | 5 | 0.088±0.048 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-ssm-atkadaptive_cosine` | 5 | 0.067±0.056 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-ssm-atkgaussian_noise` | 5 | 0.089±0.046 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-ssm-atklabel_flip` | 5 | 0.100±0.041 | 0 | — | — | |

| `temporal-t1b-pipeline_geomed-ssm-atksign_flip` | 5 | 0.057±0.053 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-cnn_lstm-atkadaptive_cosine` | 5 | 0.229±0.071 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-cnn_lstm-atkgaussian_noise` | 5 | 0.343±0.084 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-cnn_lstm-atklabel_flip` | 5 | 0.312±0.053 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-cnn_lstm-atksign_flip` | 5 | 0.319±0.053 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-snn-atkadaptive_cosine` | 5 | 0.104±0.020 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-snn-atkgaussian_noise` | 5 | 0.106±0.011 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-snn-atklabel_flip` | 5 | 0.092±0.017 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-snn-atksign_flip` | 5 | 0.084±0.045 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-ssm-atkadaptive_cosine` | 5 | 0.028±0.027 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-ssm-atkgaussian_noise` | 5 | 0.083±0.042 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-ssm-atklabel_flip` | 5 | 0.113±0.051 | 0 | — | — | |

| `temporal-t1b-pipeline_krum-ssm-atksign_flip` | 5 | 0.057±0.053 | 0 | — | — | |


**Total: 24 templates com |Δ|>0.05** (24 linhas; algumas aparecem em duplicata
estrutural no t3tau). Resumo na tabela de divergências abaixo e discussão em §4.

## 2. Claims-chave do paper

| # | claim do paper | números antigos → novos (x3) | veredito |
|---|---|---|---|
| C1 | Mamba > CNN-LSTM > SNN em macro-F1 | FedAvg random: 0.758/0.620/0.161 → **0.770/0.600/0.161** | **confirma** (orden e magnitudes; CNN −0.021 dentro do ruído) |
| C1b | idem sob split cronológico | chrono FedAvg: 0.582/0.557/0.124 — **sem runs x3** (`split_mode=temporal` não existe nas baterias novas) | **não retestado** — números ficam como estão |
| C2 | pipeline completo NÃO supera geomed sozinha (adaptive_cosine) | Mamba: geomed 0.710→**0.726**, pipeline_geo 0.064→**0.051**, pipeline_krum 0.041→**0.031** | **confirma**, folga até maior |
| C3 | FRR honesto ≈ 0.969 com pipeline (Mamba, adaptive) | geo 0.9686→**0.9689**, krum 0.9676→**0.9717** (todos os rounds); skip-r1: 0.9704→0.9721 / 0.9695→0.9750 | **confirma** ao 3º decimal |
| C4 | magnitude 10× quebra fedavg, não geomed | `t4mag` fora da x3 — só os 3 seeds antigos: no_defense 0.628→0.103 (CNN) / 0.679→0.189 (Mamba) de 1×→10×; geomed segura 0.606/0.711 | **não retestado** (gap de cobertura) |
| C5 | LDP + gate incompatível | t2c τ≤0.7: FRR=1.000, adm=0.000, F1 no piso — idêntico; t3tau SSM: fronteira τ0.7↔0.8 idêntica (τ=0.8: FRR 0.967, adm 0.75; τ=1.0: FRR 0.000, adm 1.000) | **confirma** o mecanismo; **mas a constante do piso mudou** (ver §4.2) |
| C6 | Mamba indiferente a α (flat em heterogeneidade) | clean FedAvg SSM: α0.1=0.798, α0.5=0.770, α1.0=0.757, iid=0.770 → range 0.041 | **confirma** em clean (pareado n.s.); **nuance**: sob adaptive_cosine α0.1 < iid (0.687 vs 0.763, p=0.020) — agora detectável com n=9 |

Notas C5 — valores de piso impressos mudam se a tabela for regenerada (o gate segue rejeitando
100% dos clientes; o que muda é o macro-F1 do modelo não-treinado, que depende do RNG do torch):

| célula (piso, não-treinado) | old | x3 |
|---|---|---|
| CNN-LSTM t2c (qualquer σ, τ≤0.7) | 0.054±0.061 | 0.024±0.038 |
| SNN t2c | 0.009±0.007 | 0.040±0.042 |
| Mamba t2c | 0.033±0.040 | 0.033±0.029 |
| Mamba t3tau τ=0.8 σ=0.25 | 0.082±0.068 | 0.016±0.040 |
| Mamba t3tau τ=0.8/1.0 σ≥1.0 | 0.121±0.001 | 0.088–0.121 (com divergências novas, §4.1) |

## 3. Wilcoxon pareado por seed

Piso teórico two-sided (todas as diferenças do mesmo sinal): n=5 → **0.0625** (era o teto do
paper); n=9 → **0.0039**; n=15 → **6.1e-5**.

| comparação (bateria) | n | Δ médio | sinais +/−/0 | p exato |
|---|---|---|---|---|
| x3 t1a FedAvg: Mamba vs CNN-LSTM | 15 | +0.171 | 15/0/0 | **6.1e-5** |
| x3 t1a FedAvg: Mamba vs SNN | 15 | +0.610 | 15/0/0 | **6.1e-5** |
| x3 t1a FedAvg: CNN-LSTM vs SNN | 15 | +0.439 | 15/0/0 | **6.1e-5** |
| old t1a FedAvg: os 3 pares acima | 5 | — | 5/0/0 | 0.0625 (piso) |
| x3 t1b adaptive: geomed vs pipeline_geo, Mamba | 15 | +0.675 | 15/0/0 | **6.1e-5** |
| x3 t1b adaptive: geomed vs pipeline_krum, Mamba | 15 | +0.695 | 15/0/0 | **6.1e-5** |
| x3 t1b adaptive: geomed vs pipeline_geo, CNN | 15 | +0.188 | 14/1/0 | 3.0e-4 |
| x3 t1b adaptive: geomed vs pipeline_geo, SNN | 15 | +0.067 | 14/1/0 | 1.2e-4 |
| old t1b adaptive: geomed vs pipeline_geo (qualquer modelo) | 5 | — | 5/0/0 | 0.0625 (piso) |
| x3 t1b adaptive: geomed vs no_defense, Mamba | 15 | −0.013 | 5/10/0 | 0.188 (n.s.) |
| x3 t1b adaptive: geomed vs no_defense, CNN | 15 | −0.020 | 8/7/0 | 0.489 (n.s.) |
| x3 t1a clean: FedAvg vs pipeline_krum, CNN | 15 | +0.123 | 14/1/0 | 1.2e-4 |
| x3 t1a clean: FedAvg vs pipeline_krum, Mamba | 15 | +0.572 | 15/0/0 | **6.1e-5** |
| x3 t1a clean: centralizado vs FedAvg, CNN | 15 | −0.018 | 7/8/0 | 0.639 (n.s.) |
| x3 t1a clean: centralizado vs FedAvg, Mamba | 15 | +0.078 | 11/4/0 | 0.0026 |
| x3 t1a clean: FedAvg vs local, CNN | 15 | +0.308 | 15/0/0 | **6.1e-5** |
| x3 t2a clean Mamba FedAvg: α0.1 vs α1.0 | 9 | +0.041 | 7/2/0 | 0.164 (n.s.) |
| x3 t2a clean Mamba FedAvg: α0.1 vs iid | 9 | +0.028 | 6/3/0 | 0.203 (n.s.) |
| x3 t2a clean Mamba FedAvg: α1.0 vs iid | 9 | −0.013 | 3/6/0 | 0.250 (n.s.) |
| x3 t2a clean CNN FedAvg: α0.1 vs α1.0 | 9 | −0.150 | 1/8/0 | 0.0078 |
| x3 t2a clean CNN FedAvg: α0.1 vs iid | 9 | −0.184 | 0/9/0 | 0.0039 |
| x3 t2a Mamba FedAvg α0.1: clean vs adaptive | 9 | +0.111 | 7/2/0 | 0.0195 |
| x3 t2a Mamba FedAvg α1.0: clean vs adaptive | 9 | +0.013 | 5/4/0 | 0.301 (n.s.) |
| x3 t2a Mamba adaptive: α0.1 vs iid | 9 | −0.075 | 1/8/0 | 0.0195 |
| x3 t2a Mamba adaptive: α0.1 vs α1.0 | 9 | −0.057 | 1/8/0 | 0.0391 |
| x3 t2a CNN adaptive: α0.1 vs iid | 9 | −0.097 | 1/8/0 | 0.0547 (marginal) |
| x3 t2b Mamba no_defense adaptive: frac10% vs 20% | 9 | +0.025 | 8/1/0 | 0.0117 |
| x3 t2c Mamba pipeline_krum σ0.25 vs σ2.0 (τ=0.3) | 9 | 0.000 | 0/0/9 | n/a (idênticos — piso) |
| x3 t2d CNN pipeline_krum m4 vs m7 | 9 | −0.010 | 4/5/0 | 0.820 (n.s.) |
| x3 t2e SNN ts4 vs ts16 (clean) | 9 | +0.072 | 9/0/0 | 0.0039 |
| old t2a Mamba FedAvg α0.1 vs iid | 3 | +0.018 | 1/2/0 | 1.000 (n=3 não decide nada) |

Leitura: a bateria x3 **sai do piso p=0.0625** — as comparações centrais do paper
(ordenação de modelos, geomed > pipeline) agora têm p ≈ 6e-5 a 3e-4. Os efeitos que eram
"n.s. por falta de n" continuam n.s. onde o efeito é realmente ~0 (α no Mamba clean,
geomed vs no_defense sob adaptive) — ou seja, o teste ganhou poder sem inverter conclusões.

## 4. Anomalias e alertas

### 4.1 Runs divergentes (`final.diverged`, macro-F1=0 marcador)

- **old**: 6 arquivos, todos t3tau Mamba σ=1.0, τ∈{0.8,1.0}, seed 0 — são 2 configurações
  em triplicata (fonte t0p3/t0p5/t0p7), como a própria nota do paper reconhece.
- **ssm_x3**: 6 arquivos — `t3tau-{tau0p8,tau1p0}-t2c-…-sg1p0-t0p3` seeds **0 e 3**, e
  `sg2p0-t0p3` seed **3**. Ou seja: a divergência cresceu — σ=1.0 agora diverge em 2 seeds
  e σ=2.0 (que não divergia) ganhou uma. Em n=9 cada célula afetada fica com 7–8 seeds.
- `runs_x3` (CNN/SNN): **zero** divergências. Nenhum run novo sem `macro_f1`, sem NaN,
  sem `rounds` incompletas. Cobertura 1500+1065 = 100% dos manifestos.

### 4.2 O piso "modelo não-treinado" mudou de valor

Em todo run onde o gate rejeita 100% dos clientes (t2c LDP, τ≤0.7), macro-F1 = score do
modelo na inicialização. Essa constante **mudou entre builds**: CNN 0.054→0.024,
SNN 0.009→0.040, Mamba 0.033→0.033. O mecanismo é idêntico (FRR=1.000, adm=0.000,
30 rounds sem atualizar); só o RNG de init do torch 2.9.1 desenha outros pesos.
Efeito: se `tab_ldp_curve`/`tab_ldp_tau_extended` forem regeneradas com x3, os valores de
macro-F1 mudam ~±0.03 sem mudança de interpretação — mas hoje os dois lados misturam
constantes de builds diferentes.

### 4.3 Mudança de recipe `class_weighting` — quantificação por modelo

Única diferença de config old↔x3: braço **centralized** passou de `none` para
`effective_num` (todos os demais braços já eram `effective_num` nos dois lados — a
atenção da tarefa se confirma exatamente e só nesse braço). Efeito no macro-F1 do teto:

| modelo | old `none` (n=5) | matched `eff_num` env antigo | x3 `eff_num` (n=15) | Δ total |
|---|---|---|---|---|
| CNN-LSTM | 0.666±0.053 | 0.512 (footnote do paper) | 0.582±0.100 | −0.084 |
| SNN | 0.700±0.035 | 0.595 (`runs_snn_fixed`, n=5) | 0.586±0.085 | −0.114 |
| Mamba | 0.916±0.025 | 0.827 (footnote do paper) | 0.849±0.049 | −0.068 |

Isolando ambiente vs recipe: para SNN temos as três pernas — recipe custa −0.105
(0.700→0.595 no env antigo) e o env novo custa só −0.009 adicionais (0.595→0.586).
Ou seja, **o desvio dos tetos é troca de recipe intencional, não regressão**. O paper já
conhecia os valores matched (footnote de `tab_temporal_frame`) e a x3 confirma a ordem de
grandeza (CNN/Mamba divergem ~0.06–0.07 dos valores matched antigos — ruído de env/build).

Curiosidade: com `effective_num` o teto CNN-LSTM (0.582) ≈ SNN (0.586) — a distância
CNN≫SNN dos tetos publicados era em parte artefato da recipe sem pesos.

### 4.4 Reprodução bitwise em ~29% dos runs de overlap

247 dos 855 run_ids compartilhados têm `macro_f1` **bit-idêntico** entre old e x3
(timestamps provam re-execução real, jul→out). Concentram-se em (a) runs Mamba —
pipeline SSM é determinística de ponta a ponta nessas células — e (b) runs pregados no
piso pelo gate. Consequência boa: os números velhos carregam literalmente para a nova
bateria nesses templates; o Δ vem inteiro das seeds 5–14 (t1) / 3–8 (t2).

### 4.5 `config_hash` stale nos manifestos x3 (pegadinha confirmada)

`manifest_x3.jsonl`: 1500 linhas, 1500 run_ids únicos, **só 132 config_hash únicos** —
cada hash cobre as 15 ou 9 seeds do seu template. `manifest_ssm_x3.jsonl`: 1065 linhas,
99 hashes. Quem deduplicar/agrupar por hash destrói o fator seed. Correto: `run_id`.
(Verificado nos payloads: `config.manifest.config.seed` e `federation.seed` gravam o
valor real de cada run — o conteúdo do resultado está certo; só a chave hash é falsa.)

### 4.6 Efeito ambiente ≈ 0

Δ pareado nas seeds comuns (mesmas seeds, env diferente), sobre 230 templates sem mudança
de recipe: mediana **0.000**, média −0.003; só 13 templates passam |Δ|>0.05 e todos com
sd pareado grande. O ambiente RTX5090/torch2.9.1 não move o macro-F1 de forma sistemática.
Alertas menores: `git_commit` = "unknown" em 100% dos x3 (rastreabilidade) e o env
antigo era heterogêneo (5 GPUs, 5 builds de torch) — o que infla um pouco a variância
"entre seeds" da bateria antiga.

### 4.7 Regime real nas células flagged não-recipe

Todas as divergências >0.05 fora do braço centralized são **variância de seed exposta
pela ampliação**, não mudança de mecanismo — ver distribuição por seed em §5.2. Casos:

- `t2d-pipeline_krum-cnn_lstm-m4/m7` (−0.12/−0.08): old [0.32–0.34] apertado; novas 9 seeds
  espalham 0.02–0.41. A "robustez a m" do comitê era ilusão de n=3 — o PBFT committee
  size m{4,7} é bimodal sob adaptive_cosine. **Reescrever a leitura do bloco t2d.**
- `t2a-pipeline_krum-ssm-{a0p1,a1p0}-clean` (−0.066/−0.071): seeds 0–2 bit-idênticas;
  seeds 3–8 caem para 0.09–0.25. Média antiga era otimista.
- `t2e-fidelity_only-snn` (+0.062): old tinha 2 seeds no piso (0.010, 0.003); novas 9
  seeds ficam ~0.09–0.14. Média antiga era pessimista.
- `t1b-pipeline_krum-cnn_lstm-label_flip` (+0.082): old tinha seed4=0.005 arrastando
  (0.287); novo range 0.28–0.51 (0.368).
- `t2b-no_defense-cnn_lstm` frac2 adaptive (+0.068) e sign_flip frac1/frac2 (+0.055):
  mais seeds sobem a média — n=3 subestimava o FedAvg sem defesa.
- `t3tau` SSM: 6 células flagged são a mesma família — geomed+gaussian ±0.06 e
  σ=0.25-floored ±0.066 (τ∈{0.8,1.0}); seeds compartilhadas bit-idênticas, Δ vem das
  seeds novas.

## 5. Conclusão

### 5.1 Os números seguram?

**Seguram em sinal e em ranking; mudam de magnitude em algumas células; nada muda de sinal.**
As três tabelas centrais (reference frame, attack×defense, defense decomposition) mudam
pouco: ordenação Mamba > CNN-LSTM > SNN intacta, geomed-al≈no_defense≫pipeline intacta,
FRR 0.969 intacto, LDP+gate incompatível intacto, α-flat do Mamba intacto em clean.

Duas lacunas de cobertura: **chrono/temporal e magnitude não foram re-executados** — os
números publicados para eles seguem apoiados em n=5/n=3 antigos. E o bloco t3tau só ganhou
seeds novas para Mamba.

### 5.2 O que precisaria atualizar se as tabelas forem regeneradas com x3

| tabela | bloco | o que muda |
|---|---|---|
| `tab_reference_frame` | t1a | tetos centralized: 0.666→0.582 / 0.700→0.586 / 0.916→0.849 (recipe novo — rotular); FedAvg/FedProx/pipelines ±0.02; sd encolhem ~2×; BCa CI se estreita |
| `tab_temporal_frame` | t1a±temporal | coluna random muda (inclusive tetos e FedAvg 0.620→0.600, Mamba 0.758→0.770); coluna temporal segue n=5 antiga — cuidado ao misturar n na mesma tabela |
| `tab_attack_defense` | t1b | quase tudo segura; `pipeline_krum` CNN label_flip 0.287→0.368; `krum_alone` Mamba label_flip 0.299→0.245 |
| `tab_defense_decomposition` | t1b+t2e | `fidelity_only` SNN 0.044→0.106, CNN 0.474→0.396; FRR/adm inalterados (0.969→0.969/0.972 no claim) |
| `tab_alpha_sweep` | t2a | CNN clean α0.1 0.549→0.499; Mamba pipeline_krum clean α0.1 0.272→0.206, α1.0 0.251→0.180 |
| `tab_attack_fraction` | t2b | CNN no_defense frac2 +0.068, sign_flip +0.055/0.055; pipeline_krum CNN ±0.05–0.08 |
| `tab_fedprox_mu`, `tab_coding_tsteps`, `tab_consensus_overhead` | t2e | ±0.03 suave; ts4>ts16 SNN agora p=0.004 |
| `tab_ldp_curve` + `tab_ldp_tau_extended` | t2c+t3tau | macro-F1 do piso muda (§4.2); FRR/adm idênticos; 2 divergências→3 por célula σ≥1.0 (τ≥0.8); τ=0.6 SSM reforça FRR=1.000 com n=9×4σ |
| `tab_tau_sweep` | t3tau | Mamba ±0.03–0.07; CNN/SNN sem dados novos (n=3 ainda) |
| `tab_magnitude` | t4mag | **sem dados novos** — permanece n=3 |
| `tab_stats` / testes | todos | Wilcoxon sai do piso: n=15 → p_min=6.1e-5; n=9 → 3.9e-3; Friedman/Nemenyi ganham df |

### 5.3 Recomendações

1. Republicar as células t1/t2 com as médias x3 (n=15/9) — mesmas conclusões, sd ~2× menor.
2. Antes de fechar o paper, estender a x3 a `temporal-*`, `t4mag` e `t3tau` CNN/SNN —
   são os três pontos onde as claims do paper ainda dependem de n=3–5.
3. Rotular a linha "Centralized ceiling" com a recipe (`unweighted` vs `effective_num`)
   ou adotar o valor matched (x3) — hoje a tabela mistura recipes.
4. Reler a conclusão do t2d: `m` do committee não é robusto como os 3 seeds sugeriam.
5. Ao agregar: usar `run_id` como chave (config_hash do manifesto x3 é stale) e excluir
   `final.diverged` das médias (paper já faz isso).

---

# PARTE 2 — cobertura completa

Segunda rodada de análise, depois da bateria `runs_gap_x3/` (1980 arquivos) que fechou as
lacunas apontadas na Parte 1: `temporal-*` seeds 5–14, `t3tau` seeds 3–8 (CNN/SNN/SSM),
`t4mag` seeds 3–8 e as 8 variantes `t2e` restantes seeds 3–8. **Agora todos os 511
templates da bateria estão na profundidade completa**: n=15 em t1/temporal, n=9 em
t2/t3tau/t4mag/t2e.

## P0. Fontes, casamento e método (atualizado)

| bateria | diretório | arquivos | seeds | uso |
|---|---|---|---|---|
| antiga | `runs/` | 1851 úteis | t1/temporal: 0–4, t2/t3/t4: 0–2 | números publicados + metade antiga das células novas |
| SNN corrigida | `runs_snn_fixed/` | 440 | t1: 0–4 | referência de recipe |
| x3 | `runs_x3/` | 1500 | t1: 0–14, t2: 0–8 | CNN-LSTM + SNN |
| x3 SSM | `runs_ssm_x3/` | 1065 | t1: 0–14, t2/t3tau: 0–8 | Mamba/SSM |
| gap x3 | `runs_gap_x3/` | 1980 | temporal: 5–14, t3tau/t4mag/t2e: 3–8 | fechamento de cobertura |

- Banco agregado: 6830 arquivos de run (+6 probes de latência em `runs/`) → **5535 run_ids
  únicos**, preferência de fonte em colisão `gap > ssm_x3 > x3 > runs` (a x3 re-executou as
  seeds 0–4/0–2 compartilhadas; as cópias antigas foram preteridas). Nenhum run ficou sem
  `final.macro_f1`.
- `config_hash` continua **não sendo** chave: nos payloads x3/gap ele é stale por convenção
  (verificado: `experiments/gen_gap_x3.py` copia o hash da linha-semente). Para as
  triplicatas `t3tau` (fontes `-t0p3/-t0p5/-t0p7` do grid t2c) a deduplicação usou a chave
  semântica `(modelo, sigma, tau, seed)` — dentro de cada bateria as três cópias são
  bit-idênticas; entre baterias as mesmas seeds diferem (re-execução), então a preferência
  de fonte acima decide qual cópia entra.
- Ambientes: `runs/` = mix RTX 3060/3090/4080, torch 2.4–2.13; `runs_x3`/`runs_ssm_x3` =
  RTX 5090 torch 2.9.1; `runs_gap_x3` = RTX 4090 torch 2.9.1. Células temporais misturam
  ambientes por construção (seeds 0–4 antigas + 5–14 novas); sensibilidade seeds 5–14 em
  §P1.4.
- Divergidos: 18 linhas com `final.diverged` = **6 células semânticas** do grid LDP SSM
  (`τ∈{0.8,1.0} × σ∈{1.0,2.0}`, seeds {0,3}) — excluídas das médias, contadas à parte.
- Verificação de `config_hash` canônico (sha256 do JSON canônico do config): `runs/`
  reproduz **66.7%** (1231/1845) — exatamente o "67%" do abstract; corpus completo =
  26.4% porque as linhas x3 carregam hash stale por convenção documentada.

## P1. Claim chrono/temporal — n=15

### P1.1 Frame de referência (t1a_temporal vs t1a)

`old` = n=5 publicado; `new` = n=15 (random = runs_x3/runs_ssm_x3; temporal = runs/ s0–4
+ gap s5–14). `p` = Wilcoxon pareado por seed (two-sided, exato).

| braço | modelo | random n5→n15 | temporal n5→n15 | Δ novo | Δ% | p (n=15) |
|---|---|---|---|---|---|---|
| centralized | CNN | 0.666→0.582±0.100 | 0.522→0.522±0.104 | −0.060 | −10.4% | 0.121 |
| centralized | SNN | 0.700→0.586±0.085 | 0.218→0.236±0.034 | −0.350 | −59.7% | 6.1e-5 |
| centralized | SSM | 0.916→0.849±0.049 | 0.728→0.724±0.047 | −0.125 | −14.7% | 1.2e-4 |
| local | CNN | 0.298→0.292±0.028 | 0.270→0.270±0.028 | −0.022 | −7.6% | 1.2e-3 |
| local | SNN | 0.130→0.129±0.008 | 0.106→0.106±0.006 | −0.023 | −17.6% | 6.1e-5 |
| local | SSM | 0.202→0.197±0.013 | 0.159→0.161±0.011 | −0.036 | −18.2% | 6.1e-5 |
| fedavg | CNN | 0.620→0.600±0.048 | 0.557→0.596±0.081 | −0.004 | −0.6% | 0.89 |
| fedavg | SNN | 0.161→0.161±0.021 | 0.124→0.128±0.020 | −0.033 | −20.6% | 6.1e-5 |
| fedavg | SSM | 0.758→0.770±0.037 | 0.582→0.590±0.048 | −0.180 | −23.4% | 6.1e-5 |
| fedprox | CNN | 0.622→0.603±0.056 | 0.557→0.587±0.073 | −0.016 | −2.7% | 0.169 |
| fedprox | SNN | 0.163→0.163±0.020 | 0.129→0.130±0.019 | −0.033 | −20.4% | 6.1e-5 |
| fedprox | SSM | 0.747→0.751±0.030 | 0.584→0.585±0.050 | −0.165 | −22.0% | 6.1e-5 |
| pipeline_krum | CNN | 0.495→0.476±0.069 | 0.343→0.401±0.069 | −0.075 | −15.7% | 6.7e-3 |
| pipeline_krum | SNN | 0.125→0.126±0.018 | 0.110→0.116±0.015 | −0.010 | −7.6% | 0.151 |
| pipeline_krum | SSM | 0.172→0.198±0.074 | 0.132→0.149±0.040 | −0.049 | −24.9% | 0.022 |
| pipeline_geomed | CNN | 0.475→0.454±0.067 | 0.399→0.432±0.064 | −0.023 | −5.0% | 0.169 |
| pipeline_geomed | SNN | 0.140→0.136±0.019 | 0.114→0.117±0.018 | −0.018 | −13.6% | 6.1e-4 |
| pipeline_geomed | SSM | 0.138→0.156±0.048 | 0.129→0.140±0.033 | −0.016 | −10.3% | 0.121 |

### P1.2 Leituras que mudam

1. **A ordenação vira empate no topo.** Sob FedAvg temporal n=15: SSM 0.590±0.048 vs CNN
   0.596±0.081 → Δ=−0.006, p=0.56 (two-sided, n=15). Sob FedProx: 0.585 vs 0.587,
   p=0.93. A liderança do Mamba sob chrono **não é mais distinguível de zero** — o paper
   diz "the state-space model leads (0.582) ... gap narrows to 0.025"; com n=15 a margem
   é −0.006 com sinal invertido. Ambos >> SNN (Δ=+0.46, p=6.1e-5).
2. **"Every cell but one loses 7–31%" → mudou.** Com n=15: as quedas vão de −0.6%
   (FedAvg CNN, p=0.89) a −59.7% (teto SNN). Três células têm queda <7% e **não
   significativa**: FedAvg CNN (−0.6%), FedProx CNN (−2.7%), pipeline_geomed CNN
   (−5.0%); pipeline_krum SNN (−7.6%, p=0.15) e pipeline_geomed SSM (−10.3%, p=0.12)
   também ficam n.s. **A CNN-LSTM federada é insensível ao leakage** — célula-chave nova.
3. **Teto SNN:** 0.586 (recipe matched, n=15) → 0.236±0.034 temporal → queda −59.7%
   (paper: −69% de 0.700). Caveat novo: os centralizados temporais rodam
   `effective_num` mas **sem** `threshold_init` (verificado nos configs) — o matched
   completo seria 0.586→0.236.
4. **Teto centralizado não limita CNN:** temporal FedAvg CNN 0.596 > centralized 0.522
   (Δ=+0.074, p=0.095) — mantém o sinal, agora n.s. SNN e SSM mantêm teto > federado
   (p=6.1e-5).
5. **Piso local:** fedavg/fedprox/pipelines ≥ local sob temporal em todas as células
   (p≤0.02) exceto pipeline_krum SSM (Δ=−0.013, p=0.30) e pipeline_geomed SSM (−0.022,
   p=0.007 — *abaixo* do piso local). "Todo braço federado supera o local-only" segura
   para fedavg/fedprox; falha para os pipelines Mamba.
6. **Rounds-to-target (95% do teto matched):** CNN FedAvg atinge em **15/15** seeds
   (média 16.3 rounds), FedProx 15/15 (17.0) — paper dizia 3/5 em 21.0/22.3; SSM chega
   1–2/15 seeds (round ~29). SNN continua n.r. O denominador mudou: teto matched CNN
   0.582 (não 0.666).
7. **Ataque sob chrono (t1b_temporal, n=15):** geomed_alone recupera os ataques loud
   (CNN gaussian 0.059→0.541; SSM gaussian 0.076→0.512; CNN sign_flip 0.323→0.375; SSM
   sign_flip 0.227→0.274); pipeline+Krum Mamba adaptive 0.051 vs no_defense 0.553 —
   ambos os resultados reproduzem. Sob chrono, geomed_alone > pipeline_geomed em todos os
   modelos no adaptive (CNN Δ=+0.19 p=6.1e-5; SSM Δ=+0.48 p=6.1e-5; SNN Δ=+0.03
   p=0.026).

### P1.3 Per-seed FedAvg (transparência)

- CNN random: `0.566 0.597 0.583 0.611 0.708 0.580 0.616 0.533 0.535 0.547 0.640 0.606
  0.662 0.623 0.586` → temporal: `0.514 0.420 0.588 0.595 0.671 0.547 0.673 0.702
  0.501 0.625 0.692 0.657 0.645 0.576 0.534` (6 seeds sobem, 9 caem — variância domina).
- SSM random ~0.75–0.82 → temporal 0.53–0.69, queda consistente (15/15 seeds).

### P1.4 Sensibilidade de ambiente (seeds 5–14 apenas, n=10)

Subset onde temporal = gap (RTX 4090, torch 2.9.1) e random = x3 (RTX 5090, torch
2.9.1) — aproxima o pareamento de ambiente: fedavg CNN Δ=+0.022 p=0.32; SNN −0.031
p=0.002; SSM −0.183 p=0.002; ordenação ssm-vs-cnn Δ=−0.021 p=0.43. **As conclusões da
§P1.2 não dependem do env-mix das seeds 0–4.**

## P2. Sweep de magnitude (t4mag, n=9)

### P2.1 Grid completo (média±sd, n=9; gate = pooled modelos, round 1 excluído)

| escala | braço | CNN | SNN | SSM | atk score | atk adm | FRR |
|---|---|---|---|---|---|---|---|
| 1× | no_defense | 0.598±0.064 | 0.149±0.019 | 0.727±0.053 | — | — | — |
| | geomed_alone | 0.598±0.076 | 0.146±0.016 | 0.719±0.042 | — | — | — |
| | pipeline_krum | 0.240±0.065 | 0.099±0.036 | 0.031±0.029 | 0.466 | 0.672 | 0.799 |
| | pipeline_geomed | 0.282±0.125 | 0.097±0.038 | 0.059±0.034 | 0.495 | 0.661 | 0.774 |
| 2× | no_defense | 0.537±0.096 | 0.147±0.013 | 0.729±0.050 | — | — | — |
| | geomed_alone | 0.604±0.059 | 0.151±0.021 | 0.733±0.058 | — | — | — |
| | pipeline_krum | 0.145±0.067 | 0.078±0.056 | 0.050±0.047 | 0.491 | 0.639 | 0.691 |
| | pipeline_geomed | 0.236±0.114 | 0.095±0.042 | 0.053±0.045 | 0.513 | 0.636 | 0.667 |
| 5× | no_defense | 0.138±0.019 | 0.125±0.008 | 0.245±0.070 | — | — | — |
| | geomed_alone | 0.598±0.085 | 0.155±0.021 | 0.735±0.050 | — | — | — |
| | pipeline_krum | 0.057±0.049 | 0.075±0.053 | 0.049±0.034 | 0.460 | 0.651 | 0.546 |
| | pipeline_geomed | 0.146±0.094 | 0.117±0.003 | 0.062±0.033 | 0.486 | 0.666 | 0.570 |
| 10× | no_defense | 0.103±0.025 | 0.119±0.005 | 0.205±0.088 | — | — | — |
| | geomed_alone | 0.616±0.054 | 0.153±0.014 | 0.738±0.033 | — | — | — |
| | pipeline_krum | 0.034±0.040 | 0.083±0.050 | 0.059±0.035 | 0.417 | 0.705 | 0.476 |
| | pipeline_geomed | 0.060±0.045 | 0.103±0.040 | 0.047±0.039 | 0.446 | 0.683 | 0.490 |

### P2.2 Verificação das claims

- **"10× quebra plain averaging, não geomed"** — CONFIRMA com p: no_defense s1→s10 CNN
  −0.495 (p=0.0039), SSM −0.522 (p=0.0039), SNN −0.030 (p=0.0078 — já no piso);
  geomed_alone Δ = +0.018/+0.007/+0.019, todos n.s. (p≥0.05). A s=1 a linha de controle
  reproduz t1b-adaptive dentro de ~0.03.
- **Gate loosening** — CONFIRMA: score do atacante estável (0.466→0.417 krum; gap
  discriminativo h_score−a_score: 0.608−0.466=0.142 → 0.470−0.417=0.053); admissão do
  atacante **não cai** (0.639–0.705, maior em 10×); FRR honesto 0.799→0.476 (krum) /
  0.774→0.490 (geomed).
- **"Ambos os braços gated caem ~3.9×"** — MUDA-MAGNITUDE: a CNN gated cai 4.7–7.1×
  (0.240→0.034; 0.282→0.060, ambos p=0.0039); SNN/SSM já estão no piso a s=1 e não têm
  para onde cair (SSM krum até sobe 0.031→0.059, p=0.027). Leitura honesta: "o braço
  gated CNN cai ~5–7×; os demais já estavam no piso".
- Fração de seeds no piso a s=10 (<0.121): no_defense CNN 7/9, SNN 7/9, SSM 1/9;
  pipelines 6–9/9.

## P3. Threshold LDP / τ (t3tau, n=9)

### P3.1 Sweep τ sem ruído (fonte t1b; τ=0.5 vem do T1b n=15)

Gate poola os dois braços de pipeline (n=30 em τ=0.5, n=18 nos demais), F1 por agregador:

| τ | modelo | FRR | adm | score | F1 krum | F1 geo-med |
|---|---|---|---|---|---|---|
| 0.5 | CNN | 0.691 | 0.537 | 0.528 | 0.235±0.088 | 0.420±0.140 |
| | SNN | 0.678 | 0.579 | 0.511 | 0.095±0.036 | 0.082±0.048 |
| | SSM | 0.974 | 0.870 | 0.402 | 0.031±0.025 | 0.051±0.030 |
| 0.6 | CNN | 0.425 | 0.701 | 0.491 | 0.171±0.071 | 0.274±0.185 |
| | SNN | 0.438 | 0.686 | 0.539 | 0.091±0.049 | 0.072±0.053 |
| | SSM | 0.852 | 1.000 | 0.368 | 0.054±0.042 | 0.031±0.017 |
| 0.8 | CNN | 0.082 | 0.925 | 0.394 | 0.109±0.086 | 0.584±0.124 |
| | SNN | 0.111 | 0.937 | 0.310 | 0.055±0.059 | 0.141±0.025 |
| | SSM | 0.064 | 1.000 | 0.299 | 0.056±0.035 | 0.643±0.082 |
| 1.0 | CNN | 0.000 | 0.987 | 0.374 | 0.061±0.036 | 0.599±0.076 |
| | SNN | 0.016 | 0.969 | 0.306 | 0.091±0.049 | 0.145±0.015 |
| | SSM | 0.000 | 1.000 | 0.287 | 0.056±0.035 | 0.718±0.042 |

- **CONFIRMA** "FRR cai 0.68–0.97 → ≤~0.11 em τ=0.8 → ≤0.016 em τ=1.0" (SNN fica em 0.111,
  levemente acima do "<0.09" publicado — ajuste fino) e "admissão ≥0.90 em τ≥0.8".
- **CONFIRMA** recuperação do geo-med: média sobre modelos 0.184→0.487 (paper: 0.187→
  0.472); Mamba 0.051→0.718 — geo@τ1.0 = geomed_alone em todos os modelos (Δ ≤ 0.005,
  p≥0.46, pareado s0–8). Krum continua colapsado (0.120→0.069 médio).
- Wilcoxon pareado τ0.5→1.0 no braço geo (n=9): CNN +0.133 p=0.012; SNN +0.060 p=0.0078;
  SSM +0.661 p=0.0039.

### P3.2 Grid LDP estendido (t2c τ≤0.7 + t3tau τ∈{0.6,0.8,1.0}, dedup semântico)

| τ | σ | FRR | adm | CNN | SNN | SSM |
|---|---|---|---|---|---|---|
| ≤0.7 | todas | 1.000 | 0.000 | 0.024 | 0.040 | 0.033 |
| 0.8 | 0.25 | 0.967 | 1.000 | 0.047 | 0.029 | 0.016 |
| 0.8 | 0.5 | 0.967 | 1.000 | 0.050 | 0.025 | 0.016 |
| 0.8 | 1.0 | 0.967 | 0.891 | 0.050 | 0.024 | 0.088 (7/9, 2div) |
| 0.8 | 2.0 | 0.967 | 0.788 | 0.051 | 0.023 | 0.094 (8/9, 1div) |
| 1.0 | 0.25/0.5 | 0.000 | 1.000 | 0.047–0.050 | 0.025–0.029 | 0.016 |
| 1.0 | 1.0 | 0.000 | 1.000 | 0.050 | 0.024 | 0.121 (7/9, 2div) |
| 1.0 | 2.0 | 0.000 | 1.000 | 0.052 | 0.023 | 0.121 (8/9, 1div) |

- **Fronteira τ=0.7↔0.8 CONFIRMADA com n=9** — idêntica à n=3: mecanismo, não amostragem.
- **Piso de inicialização mudou para CNN/SNN**: 0.054→0.024 e 0.009→0.040 (SSM 0.033
  estável) — avaliação do init depende do ambiente/torch; a claim "não aprende" segura
  (todos < 0.121).
- Divergências: 6 células semânticas (SSM σ∈{1,2} em τ∈{0.8,1.0}; seeds 0 e 3 — a seed 3
  diverge nas duas baterias, reproduzível entre envs). Paper dizia "6 runs, seed 0" →
  corrigir para "6 células, seeds {0,3}".
- Admissão do atacante em τ≥0.8: 0.788–1.000 (paper: 0.774–1.000) — CONFIRMA.
- Nuance per-seed: em τ≥0.8, SSM σ≥1.0 concentra no piso do preditor trivial 0.121
  (7–9 seeds), σ≤0.5 fica abaixo do init (0.001–0.121) — "floor" tem duas camadas.

## P4. Robustez a M (t2d, n=9) — reescrita necessária

Per-seed pipeline_krum adaptive:

| M | CNN | SNN | SSM |
|---|---|---|---|
| 4 | 0.210±0.121 — `0.155 0.195 0.326 0.409 0.098 0.147 0.021 0.239 0.295` | 0.103±0.036 | 0.023±0.018 |
| 7 | 0.220±0.097 — `0.258 0.183 0.177 0.043 0.322 0.115 0.262 0.320 0.299` | 0.106±0.032 | 0.024±0.020 |
| 10 (t1b) | 0.235±0.088 (n=15) | 0.095±0.036 | 0.031±0.025 |

- Fração de seeds abaixo do piso trivial 0.121: CNN 2/9 (m4), 2/9 (m7), 4/15 (m10); SNN
  7/9, 7/9, 9/15; SSM 9/9, 9/9, 15/15. Abaixo de 0.30: CNN 7/9/11, SNN 9/9/15,
  SSM 9/9/15.
- Wilcoxon pareado m4-vs-m7, m4-vs-m10, m7-vs-m10 (n=9): todos p≥0.25 — flat confirmado.
- **Leitura honesta substituta** para "robusto a m≤7": *o tamanho do comitê não muda
  nada porque o braço já está colapsado em todo M* — a CNN fica bimodal em 0.02–0.41
  (7/9 seeds <0.30), SNN ~0.10, Mamba ~0.025. O sweep mede variância de seed de um braço
  quebrado, não robustez do comitê. Overhead confirmado: 24/84/180 msgs, p50
  113.4/145.4/167.8 ms, p95 158.6/199.5/221.2 ms.

## P5. t2e completo (n=9)

### P5.1 SNN coding × t_steps (clean / adaptive)

| coding | T | clean | adaptive | SynOps/inf |
|---|---|---|---|---|
| rate | 4 | 0.122±0.004 | 0.121±0.001 | 422 827 |
| rate | 8 | 0.121±0.000 | 0.121±0.001 | 849 521 |
| rate | 16 | 0.121±0.000 | 0.121±0.005 | 1 693 432 |
| delta | 4 | 0.204±0.037 | 0.192±0.020 | 47 681 |
| delta | 8 (t1) | 0.161±0.021 (n=15) | 0.152±0.018 | 70 079 |
| delta | 16 | 0.132±0.016 | 0.121±0.008 | 133 153 |

- Wilcoxon pareado (n=9): delta T4>T8 (+0.046 clean/+0.044 adaptive, p=0.0039) e
  T8>T16 (−0.026/−0.028, p=0.0039) — a ordenação T4>T8>T16 agora é estatística, não só
  descritiva. Rate é plano em T (p=1.0) e preso no piso trivial.
- **CONFIRMA** "delta < rate em energia e só delta sai do piso"; SynOps a n=15/9
  desloca ~2% do publicado.

### P5.2 SSM ts4/ts16/codingrate — configs degeneradas

`t2e-fedavg-ssm-ts4`, `-ts16` e `-codingrate` têm **configs bit-idênticos** (o modelo
SSM não tem `t_steps`/`coding` — os três templates são a mesma linha com seeds iguais).
54 runs (3 templates × 2 ataques × 9 seeds) medem a mesma coisa: clean 0.764/0.765/0.764,
adaptive 0.729/0.728/0.726. São "peso morto" da bateria — não um sweep real.

### P5.3 FedProx μ (n=9) e fidelity_only

| μ | CNN Δ | SNN Δ | SSM Δ |
|---|---|---|---|
| 0.001 | +0.004 (p=0.82) | −0.000 (p=0.57) | −0.003 (p=0.20) |
| 0.1 | −0.007 (p=0.50) | +0.006 (p=0.25) | **−0.150 (p=0.0039)** |

- FedProx inerte → CONFIRMA, com uma exceção agora significativa: μ=0.1 **prejudica o
  Mamba** (0.614±0.052 vs pareado 0.7638, p=0.0039 — era "0.587 vs 0.7319" sem p).
- fidelity_only (n=9): CNN 0.396±0.110, SNN 0.106±0.034, SSM 0.064±0.038; admissão
  0.606–0.856, FRR 0.711–0.964 — o portão sozinho não basta; CONFIRMA.

## P6. Bateria estatística com os novos n (Wilcoxon central)

Pisos exatos novos: n=15 → 2⁻¹⁵ = 3.05e-5 por teste; n=9 → 3.91e-3. Pisos de família
(Holm): H2 m=24 → **7.3e-4**; H3a/H3b m=6 → **1.8e-4**; H5b m=3 → **5.9e-3** — todas as
famílias agora **resolvíveis** a α=0.05.

### P6.1 Famílias pré-registradas

| família | teste | n | resultado antigo | resultado novo |
|---|---|---|---|---|
| H1a parity | McNemar+TOST | 15 | b=400 c=5797 p<0.001, CI [−0.527,−0.426] | b=713 c=18215, χ²=16181.6, p≈0; CI [−0.465,−0.417]; **rejeitada** |
| H1b energia | BCa+Wilcoxon | 15 | ρ CI [0.006,0.008] | ρ=0.0068 [0.0063,0.0076], p_less=3.05e-5; **suportada** |
| H2 defesa | Wilcoxon+Holm 24 células | 15 | 0/24, família indecidível (floor 0.75) | 2/24 suportam (holm≤7.3e-4): Krum/CNN/gaussian e geo/CNN/gaussian (δ=+1.00, Δ+0.31); a 3ª célula antiga (Krum/SNN/label-flip δ+0.76) evaporou (δ+0.18, p=0.34). Família segue **não suportada** — e agora decidida |
| Friedman (5 braços) | seed-mean | 12 blocos | Q=20.4, 2 pares separam | Q=22.87 p=1.4e-4, CD=1.761; geomed_alone separa de ambos pipelines (holm 1.1e-4 / 1.6e-3); sensibilidade por-seed n=180: Q=271.9, 8/10 pares |
| H3a evasão | Wilcoxon+Holm 6 células | 15 | floor 0.188, "indecidível" | **6/6 células p_holm=1.8e-4, δ=+1.00 → H3a SUPORTADA** (score sign-flip ≈1.02 ≫ adaptativo ≈0.50) |
| H3b dano | Wilcoxon+Holm 6 células | 15 | 0/6, indecidível | 2/6 na direção: Krum/CNN (drop 0.242 vs 0.153, p=0.0075, holm 0.045) e Krum/SSM (0.167 vs 0.122, p=0.0083, holm 0.045); geo-mediana segue invertida (0.034<0.134) → família não suportada, mas a frase "nenhuma seed repararia" cai para o Krum |
| H4 privacidade | JT por τ | 108 | flat p=0.500 | idêntico (mecanismo, não amostragem) |
| H5a hetero | JT 6 tendências | ~42/trend | 2/6 + 2 na fronteira | **4/6** holm<0.05: FedAvg/CNN (8.1e-7), FedAvg/SNN (4.7e-7), pkrum/CNN (2.6e-5), pkrum/SNN (2.5e-3); SSM plana nos dois braços (holm 0.94/0.46) |
| H5b diferencial | Wilcoxon+Holm 3 | 9 | floor 0.375, indecidível | 1/3: SNN suporta (drop_fedavg 0.072 > drop_pipe 0.027, p=0.002, holm 0.0059, δ=+0.88); CNN invertido (0.184 < 0.257, p=0.95); SSM n.s. |

### P6.2 Testes de comparação central (headline)

| comparação | n | Δ | p |
|---|---|---|---|
| temporal FedAvg: SSM vs CNN | 15 | −0.006 | 0.56 |
| temporal FedAvg: SSM vs SNN | 15 | +0.462 | 6.1e-5 |
| temporal FedAvg: CNN vs SNN | 15 | +0.468 | 6.1e-5 |
| temporal vs random FedAvg: CNN / SNN / SSM | 15 | −0.004 / −0.033 / −0.180 | 0.89 / 6.1e-5 / 6.1e-5 |
| temporal vs random centralized: CNN / SNN / SSM | 15 | −0.060 / −0.350 / −0.125 | 0.12 / 6.1e-5 / 1.2e-4 |
| temporal centralizado vs FedAvg (CNN) | 15 | −0.074 (fed > cen) | 0.095 |
| temporal FedAvg/FedProx/pipes vs local | 15 | +0.33/+0.32/+0.13..−0.02 | ≤0.02 exceto pipes-SSM |
| t4mag s1→s10 no_defense CNN/SNN/SSM | 9 | −0.495/−0.030/−0.522 | 0.0039/0.0078/0.0039 |
| t4mag s1→s10 geomed (3 modelos) | 9 | +0.007..+0.019 | n.s. |
| t3tau geo τ0.5→1.0 CNN/SNN/SSM | 9 | +0.133/+0.060/+0.661 | 0.012/0.0078/0.0039 |
| t2d m4↔m7↔m10 (9 pares) | 9 | ≤0.01 | todos n.s. |
| t2e delta T4↔T8↔T16 SNN | 9 | +0.046/−0.026 | 0.0039/0.0039 |
| fedprox μ0.1 SSM vs FedAvg | 9 | −0.150 | 0.0039 |

Ordenação aleatória (t1a FedAvg, n=15): SSM 0.770 > CNN 0.600 > SNN 0.161 — intacta.
Ordenação temporal: **CNN 0.596 ≈ SSM 0.590 > SNN 0.128** — Mamba já não "lidera".

## P7. Quadro de claims

Legenda: **CONFIRMA** (direção e magnitude mantidas), **MUDA-MAGNITUDE** (mesmo sinal,
número precisa atualizar), **MUDA-SINAL** (direção/ordenação muda), **SEM-EVIDÊNCIA**
(cobertura não decide). Locais: `main.tex` (abstract), `07_results.tex`, `08_discussion.tex`
(números de linha aproximados).

| # | claim (local) | veredito | evidência |
|---|---|---|---|
| C1 | Abstract: chrono Mamba "leads" 0.58, CNN 0.56, SNN 0.12 | **MUDA-SINAL** | 0.590 vs 0.596 — empate estatístico (Δ−0.006, p=0.56); CNN nominalmente à frente |
| C2 | Abstract: random 0.76/0.62/0.16 é teto inflado | MUDA-MAGNITUDE | 0.770/0.600/0.161; CNN-FedAvg não cai com o corte (insensível a leakage) |
| C3 | Abstract: SNN a 0.7% da energia | CONFIRMA | ρ=0.0068, BCa [0.0063,0.0076], n=15 |
| C4 | Abstract: só CNN-LSTM GPU p99 (3.3 ms, 82%) cabe 4 ms | SEM-EVIDÊNCIA-NOVA | medição única (latency-*.json em runs/); não re-executada — mantida por design |
| C5 | Abstract: pipeline não supera averaging/geomed — "compra auditabilidade" | CONFIRMA | n=15: geomed_alone > pipelines em toda célula; H2 decidido: só 2/24 células a favor |
| C6 | Abstract: LDP utilizável incompatível com o gate | CONFIRMA | τ≤0.7 FRR=1.000/adm=0/floor em n=9; τ≥0.8 adm≥0.79 |
| C7 | Abstract: magnitude 10× quebra averaging, não geomed; FRR 0.80→0.47, admissão não cai | CONFIRMA | CNN 0.598→0.103 p=0.0039; geomed Δ≤0.019 n.s.; FRR 0.799→0.476; adm 0.672→0.705 |
| C8 | Abstract: 1311+534 runs; 67% hashes canônicos | CONFIRMA c/ ressalva | runs/ reproduz 66.7%; corpus completo 26.4% (hash stale por convenção); corpus total agora 5535 run_ids |
| C9 | §7.1: "ordering survives, margin narrows 0.138→0.025" | **MUDA-SINAL** | gap = −0.006 (invertido, n.s.); FedProx −0.002 |
| C10 | §7.1: "every cell but one loses 7–31%" | MUDA-MAGNITUDE | faixa −0.6%…−59.7%; 5 células n.s.; CNN federada imune a leakage |
| C11 | §7.1: teto spiking 0.700→0.218 (−69%) | MUDA-MAGNITUDE | matched 0.586→0.236 (−59.7%); caveat: lado temporal sem threshold_init |
| C12 | §7.1: teto centralizado não limita CNN-LSTM | CONFIRMA | temporal fed 0.596 > cen 0.522 (p=0.095, n.s.); random 0.600 > 0.582 |
| C13 | §7.1: probe GBT 0.692/0.533 etc. | SEM-EVIDÊNCIA-NOVA | probe fora da bateria (paper/generated/gbt_baseline.json) |
| C14 | §7.1: rounds-to-target 21.0/22.3 em 3/5 seeds; demais n.r. | MUDA-MAGNITUDE | CNN FedAvg/FedProx 15/15 em 16.3/17.0 (denominador = teto matched 0.582); SSM 1–2/15 |
| C15 | §7.2: robust agg recupera loud attacks sob chrono; pipeline+Krum Mamba 0.028 | CONFIRMA | geomed gaussian CNN 0.541, SSM 0.512; pkrum-SSM adaptive 0.051 vs no_defense 0.553 |
| C16 | §7.3 H1a: paridade rejeitada (b=400,c=5797; CI [−0.527,−0.426]) | CONFIRMA (mais forte) | b=713, c=18215, CI [−0.465,−0.417] |
| C17 | §7.3 H1b: ρ≈6.7e-3, CI<1, ~150× mais barato | CONFIRMA | ρ=0.0068 [0.0063,0.0076]; SynOps delta T8 70 079 |
| C18 | §7.4: latências/p99 e única config dentro do envelope | SEM-EVIDÊNCIA-NOVA | probes únicos, env antiga; mantidos |
| C19 | §7.5: decomp — geomed alone preserva (0.631/0.710), Krum alone colapsa 0.058–0.121 | CONFIRMA c/ update | geomed 0.609/0.726; Krum alone agora 0.043–0.100; fidelity_only 0.396/0.106/0.064 |
| C20 | §7.5: Friedman n=12 separa geomed dos dois pipelines; sensibilidade 8/10 | CONFIRMA | Q=22.87 CD=1.761 (n=12); sensibilidade n=180, 8/10 |
| C21 | §7.5: "0/24 registra orçamento, não derrota" | MUDA-MAGNITUDE | família agora decidível: 2/24 holm<0.05 (só gaussian/CNN) |
| C22 | §7.5: t2b "accuracy falls as more defect; Mamba resiste adaptive em toda fração" | CONFIRMA | no_defense adaptive SSM 0.733–0.758 em f∈{0.1,0.2,0.3}; sinais degradam com f |
| C23 | §7.6: FRR 0.68–0.97→<0.09→≤0.014; adm ≥0.90 em τ≥0.8 | CONFIRMA c/ ajuste | SNN τ0.8 FRR=0.111 (acima do <0.09); resto dentro |
| C24 | §7.6: geo-med recupera 0.187→0.472, Mamba →0.689≈geomed-alone | CONFIRMA | 0.184→0.487; SSM →0.718 vs alone 0.726 (Δ 0.008) |
| C25 | §7.6: Krum colapsa 0.125→0.082 | CONFIRMA | 0.120→0.069 |
| C26 | §7.7 H3a: evasão real mas "teste não sobrevive a correção; n=5 floor 0.188" | **MUDA-MAGNITUDE→SUPORTADA** | n=15: 6/6 células p_holm=1.8e-4; H3a vira suportada |
| C27 | §7.7 H3b: dano falha "a failure no number of seeds would repair" | MUDA-MAGNITUDE | geo-med ainda invertida; mas Krum CNN+SSM agora **a favor** (holm 0.045) — a frase é forte demais |
| C28 | §7.7: admissão 0.000→0.549 (0.844 Mamba), FRR 66–97% | CONFIRMA | adm 0.530–0.870 (Mamba 0.870); FRR 0.655–0.997 |
| C29 | §7.8: magnitude (tabela+prosa: FRR 0.801→0.469, score gap 0.132→0.019, "gated ~3.9×") | CONFIRMA c/ ajuste | FRR 0.799→0.476; gap 0.142→0.053; queda gated ~5–7× só na CNN (demais já no piso) |
| C30 | §7.9 H4/LDP: FRR=1.000 em 12 pontos; JT p=0.500; piso 0.054/0.009/0.033; abre entre τ0.7↔0.8; adm 0.774–1.000; F1 floor 0.036–0.121; 6 divergências seed 0 | CONFIRMA c/ updates | piso novo 0.024/0.040/0.033; adm 0.788–1.000; F1 0.016–0.121; divergências = 6 células em seeds {0,3} |
| C31 | §7.10 H5a: CNN 0.665→0.549, SNN 0.204→0.124, Mamba plano 0.748–0.774; "2/6 trends" | MUDA-MAGNITUDE | CNN 0.683→0.499, SNN 0.190→0.118, Mamba 0.757–0.798 plano; **4/6 trends** holm<0.05 |
| C32 | §7.10 H5b: indecidível (floor 0.375), δ −0.33/+0.78/+0.33 | MUDA-MAGNITUDE | n=9: 1/3 (SNN δ+0.88, holm 0.0059); CNN inverte sinal (δ−0.51) |
| C33 | §7.11: 180 msgs, p50 168.3, p95 231.8; "flat-to-falling" | CONFIRMA c/ reescrita | msgs/lat ~idênticos; mas o sweep não mede robustez — braço já colapsado em todo M |
| C34 | §7.11: "auditability é design claim (Ledger.verify não medido)" | CONFIRMA | sem runs de verificação; invariante estrutural |
| C35 | §7.12: CNN normal 0.976@3676, overload 0.124; SNN fault_lg rec 0.196 F1 0.040 | MUDA-MAGNITUDE | n=15: normal 0.983@3674; overload 0.089; SNN fault_lg rec 0.147, F1 0.056 |
| C36 | §7.13: delta T4 0.196 > T8 > T16; rate pior/plano; "melhor SNN ≈ ¼ do Mamba" | CONFIRMA c/ update | T4 0.204>T8 0.161>T16 0.132 (p=0.0039); 0.204/0.770 ≈ 26% |
| C37 | §7.13: FedProx inerte (maior Δ +0.007; Mamba μ0.1 −0.145 sem p) | CONFIRMA c/ update | inerte nos demais; Mamba μ0.1 agora **significativo** −0.150 p=0.0039 |
| C38 | §7.14: real-trace sanity (SNN 0.996, Mamba 1.000, CNN mais fraca) | SEM-EVIDÊNCIA-NOVA | fora da extensão X3 (runs_real, n=3) |
| C39 | §7.15: scoreboard "só H1b suportada; H2/H3/H5b indecidíveis" | MUDA-MAGNITUDE | H1a,H3b,H5b decididas; **H3a suportada**; H2 2/24; H5a 4/6; H5b 1/3 |
| C40 | §8: "Mamba é a escolha de acurácia (0.582)" | MUDA-MAGNITUDE | temporal 0.590≈CNN 0.596 — "escolha" vira empate |
| C41 | §8: "todo braço federado bate o local-only nos dois protocolos" | MUDA-MAGNITUDE | segura p/ fedavg/fedprox; pipeline_krum/geomed SSM ficam **abaixo** do local temporal |
| C42 | §8: "0.19 de leakage nas células com acurácia" | CONFIRMA c/ nuance | SSM fed −0.18; CNN fed só −0.004 — leakage é seletivo por arquitetura |
| C43 | §8: "heterogeneidade deixa o Mamba intacto" | CONFIRMA | plano 0.757–0.798 (α), JT n.s. |
| C44 | §8: "deploy the geometric median on its own" | CONFIRMA | reforçado: invariante a magnitude (n=9), recupera com τ=1.0, melhor rank Friedman |
| C45 | run-counts: "1311+534" | MUDA-MAGNITUDE | corpus = 6836 arquivos / 5535 run_ids únicos na bateria estendida |

## P8. Tabela de atualizações do paper

Colunas: local aproximado | número atual | número correto (cobertura completa) | nota.

| local | atual | correto (n=15/n=9) | nota |
|---|---|---|---|
| `main.tex:81` | "state-space model leads (0.58), reference 0.56, spiking 0.12" | SSM 0.590 / CNN 0.596 / SNN 0.128 — **empate** SSM≈CNN (Δ−0.006, p=0.56) | MUDA-SINAL: "leads" → "empata no topo"; sugerir "the two dense models tie at ~0.59" |
| `main.tex:83` | random 0.76/0.62/0.16 | 0.770/0.600/0.161 | magnitude: CNN perde o rótulo de inflada por leakage (FedAvg Δ=−0.004) |
| `main.tex:82-83` | "at 0.7% of the reference's energy" | 0.68% (ρ CI [0.63%,0.76%]) | ok, ~igual |
| `main.tex:89-91` | "FRR 0.80→0.47, admission does not fall" | 0.799→0.476 (krum) / 0.774→0.490 (geo); adm 0.672→0.705 | ok |
| `main.tex:92-93` | "1,311 + 534 runs; 67% hashes" | corpus 5535 run_ids (6836 arquivos); 66.7% em runs/, 26.4% total | atualizar contagem + qualificar o % sobre a bateria original |
| `tab_temporal_frame.tex` | coluna temporal n=5 | coluna n=15 (ver §P1.1) | todas as 18 células; sd tipicamente ±0.02–0.10 |
| `tab_reference_frame.tex` | n=5 | n=15 (Parte 1 já lista) | tetos: rotular recipe (0.666→0.582 / 0.700→0.586 / 0.916→0.849) |
| `tab_temporal_attack.tex` | n=5 | n=15 (ver §P1.2 item 7 + saída completa) | todas as células sobem ~0.01–0.05 vs n=5 |
| `07_results.tex:61-70` | "ordering survives... gap narrows 0.138→0.025" | gap = −0.006 (n.s.) | reescrever o parágrafo: empate, não lead |
| `07_results.tex:63` | "every cell but one loses 7–31%" | faixa −0.6%…−60%, 5 células n.s. | CNN federada é insensível a leakage |
| `07_results.tex:74-87` | "0.700→0.218, −69%; matched 0.595" | matched random 0.586→temporal 0.236 (−59.7%); caveat: temporal sem threshold_init | manter leitura, ajustar números |
| `07_results.tex:94-105` | "0.512 vs 0.620 / 0.522 vs 0.557" | temporal 0.522 vs 0.596 (p=0.095); matched random 0.512 vs 0.600 | claim fraca segura |
| `07_results.tex:162-166` | "21.0/22.3 em 3/5 seeds" | 16.3/17.0 em 15/15 seeds (teto matched) | atualizar |
| `07_results.tex:195-198` | geomed 0.050→0.497 / 0.075→0.483; pkrum-Mamba 0.028 | 0.059→0.541 / 0.076→0.512; 0.051 | confirma |
| `07_results.tex:232-235` | b=400 c=5797 χ²=4698.5; CI [−0.527,−0.426] | b=713 c=18215 χ²=16181.6; CI [−0.465,−0.417] | confirma, mais forte |
| `07_results.tex:240-247` | ρ≈6.7e-3, 68 606 SynOps, 21.1 µJ | ρ=0.0068, 70 079 SynOps (delta T8, n=15) | ajuste fino |
| `07_results.tex:377-395` | decomp: geomed 0.631/0.710; Krum 0.058–0.121; Mamba no-def 0.714 | geomed 0.609/0.726; Krum 0.043–0.100; no-def 0.739 | `tab_defense_decomposition.tex`/`tab_attack_defense.tex` regerar |
| `07_results.tex:397-405` | Friedman 2 pares, p_holm 4.8e-4/2.1e-3; sens. n=60 8/10 | p_holm 1.1e-4/1.6e-3; sens. n=180, 8/10, CD 0.455 | ok c/ n |
| `07_results.tex:407-424` | "0/24 floor 0.75; 3 células a favor" | família decidível: 2/24 holm≤7.3e-4 (gaussian/CNN); 3ª evaporou | reescrever |
| `07_results.tex:442-458` | τ sweep: FRR <0.09; geo 0.187→0.472; Mamba→0.689; krum 0.125→0.082 | FRR ≤0.111 em τ0.8; geo 0.184→0.487; SSM→0.718 (Δ 0.008); krum 0.120→0.069 | `tab_tau_sweep.tex` regerar (n=9) |
| `07_results.tex:493-513` | H3a "correção ata, n=5"; H3b "irreparável" | H3a suportada 6/6 (holm 1.8e-4); H3b: 2/6 a favor sob Krum | **reescrever §7.7**: evasão agora é estatística |
| `tab_adaptive_evasion.tex` | n=5 | n=15 (§P6/out_2: score 0.50/1.02; adm 0.530–0.870; FRR 0.655–0.997) | |
| `tab_magnitude.tex` | n=3 | n=9 (§P2.1) | CNN s1 0.598 (era 0.628); SSM s10 0.205; "~3.9×" → "5–7× na CNN, demais no piso" |
| `07_results.tex:539-556` | "0.628→0.103 / 0.679→0.189; adm band 0.665–0.754; score 0.465→0.415; gap 0.132→0.019" | 0.598→0.103 / 0.727→0.205; band 0.639–0.705; 0.466→0.417; gap 0.142→0.053 | |
| `tab_ldp_curve.tex`/`tab_ldp_tau_extended.tex` | n=3; piso 0.054/0.009/0.033; adm 0.774–1.000 | n=9; piso 0.024/0.040/0.033; adm 0.788–1.000; F1 0.016–0.121 | 6 células divergentes em seeds {0,3}, não só s0 |
| `07_results.tex:592-617` | "curva degenerada ... floor 0.036–0.121" | floor 0.016–0.121; duas camadas (init vs preditor-trivial) | confirma |
| `tab_alpha_sweep.tex` | n=3/5 | n=9/15 (§out_main: CNN 0.499–0.683; SNN 0.118–0.190; SSM 0.757–0.798) | |
| `07_results.tex:646-671` | "2/6 trends; 0.060/0.078 na borda; H5b floor 0.375" | 4/6 trends holm<0.05; H5b 1/3 (SNN holm 0.0059); CNN invertido | |
| `tab_consensus_overhead.tex` | M4/7/10: 0.334/0.304/0.216 CNN etc., n=3 | 0.210±0.121/0.220±0.097/0.235±0.088 (n=9/15) | reescrever leitura: variância bimodal, não robustez |
| `07_results.tex:689-695` | "flat-to-falling" | "flat porque já colapsado em todo M; 7–9/9 seeds <0.30" | |
| `tab_per_class.tex` + §7.12 | normal 0.976@3676; overload 0.124; SNN fault_lg 0.196/0.040 | 0.983@3674; 0.089; rec 0.147/F1 0.056 | |
| `tab_coding_tsteps.tex` + §7.13 | rate/delta × T a n=3 | n=9/15 (§P5.1); T4>T8>T16 com p=0.0039 | adicionar s.n. das variantes SSM degeneradas |
| `tab_fedprox_mu.tex` + §7.13 | Δs a n=3 | n=9 (§P5.3); Mamba μ0.1 −0.150 p=0.0039 | |
| `tab_stats.tex`/`tab_hypotheses.tex` | pisos 0.75/0.188/0.375; "indecidível" | pisos 7.3e-4/1.8e-4/5.9e-3; famílias decididas; H3a→suportada; H5b SNN→suportada | tabelas inteiras regerar |
| `08_discussion.tex:15-26` | "Mamba accuracy choice (0.582)" | 0.590≈CNN 0.596 — empate | ajustar frase de corner |
| `08_discussion.tex:50-61` | "every arm but one scored higher; 0.19 leakage" | 13/18 células significativas; CNN fed n.s. | ajustar |
| `08_discussion.tex:285-286` | "every federated arm beats local-only floor both protocols" | fedavg/fedprox sim; pipelines-SSM abaixo do local temporal | adicionar exceção |
| `08_discussion.tex:157-165` | "damage did not follow" | sob geo-med segue invertido; sob Krum o adaptativo já dá MAIS dano (2 células, holm 0.045) | qualificar por agregador |

## P9. Resumo executivo da Parte 2

1. **Headline chrono muda de sinal**: Mamba não lidera mais — empata com a CNN-LSTM
   (0.590 vs 0.596, n=15, p=0.56). O paper inteiro deve trocar "leads" por "tie"; a
   claim "margem estreita mas positiva" vira "margem zero".
2. **A CNN-LSTM é insensível ao split** (FedAvg Δ=−0.004, n.s.) — o efeito leakage é
   concentrado em SNN/SSM/centralized/local, não universal.
3. **Magnitude (n=9)**: a assimetria central CONFIRMA com p-valor — averaging quebra
   (p=0.0039), geomed inerte (n.s.), gate afrouxa (FRR 0.80→0.48, admissão sobe).
4. **LDP-τ (n=9)**: fronteira τ0.7↔0.8 CONFIRMADA como mecanismo; piso de init mudou
   (0.024/0.040/0.033); 6 células divergentes em seeds {0,3}.
5. **t2d**: a "robustez a m" era ilusão de n=3 — pipeline_krum fica bimodal/colapsado
   em todos os M; reescrever como "o comitê não compra acurácia porque não há acurácia
   para comprar".
6. **t2e**: delta T4>T8>T16 agora estatística; variantes SSM ts4/ts16 são configs
   idênticas (peso morto); FedProx μ0.1 prejudica Mamba significativamente.
7. **Estatística**: famílias deixaram o piso — H3a fica SUPORTADA (6/6), H3b ganha 2
   células sob Krum, H5a vai a 4/6, H5b a 1/3, H2 a 2/24; H4 continua flat por
   mecanismo. Apenas H1b era suportada — agora H1b+H3a.
8. **Nada mudou**: latência (probe único), GBT probe, real-trace, energia (ρ=0.0068),
   geomed-alone como melhor defesa, incompatibilidade LDP+gate, hash canônico da bateria
   original (66.7% ≈ "67%").
