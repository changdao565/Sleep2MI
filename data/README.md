# Data access and redistribution

No EEG recordings or participant-level derived data are distributed in this
repository. The manuscript uses the following public datasets:

| Dataset | Role in the study | Provider record and persistent identifier |
| --- | --- | --- |
| Sleep-EDF Expanded | Sleep-domain representation learning and sleep-side analyses | [PhysioNet v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/); [doi:10.13026/C2X676](https://doi.org/10.13026/C2X676) |
| CAP Sleep Database (CAPSleep) | External sleep-stage transfer and sleep-physiology analyses | [PhysioNet v1.0.0](https://physionet.org/content/capslpdb/1.0.0/) |
| EEG Motor Movement/Imagery Dataset (EEGMMIDB) | Cross-sectional MI-BCI aptitude and awake-physiology analyses | [PhysioNet v1.0.0](https://physionet.org/content/eegmmidb/1.0.0/); [doi:10.13026/C28G6P](https://doi.org/10.13026/C28G6P) |
| Cho2017 | Independent cross-sectional MI-BCI aptitude cohort and awake-domain control source | [GigaDB dataset 100295](https://doi.org/10.5524/100295) |
| Stieger2021 | Longitudinal Session 1 to Session 2 BCI-performance analysis | [figshare dataset](https://doi.org/10.6084/m9.figshare.13123148.v1) |
| WBCIC-SHU | Independent awake-rest source for the matched-source and continuation analyses | [figshare dataset](https://doi.org/10.25452/figshare.plus.22671172) |
| OpenBMI (Lee2019) | External motor-imagery donor cohort for MI-specialized comparison encoders | [GigaDB dataset 100542](https://doi.org/10.5524/100542) |
| Simultaneous EEG and fMRI Signals During Sleep from Humans | Same-subject wake-to-sleep identity analysis | [OpenNeuro ds003768 v1.0.0](https://doi.org/10.18112/openneuro.ds003768.v1.0.0) |

The links above identify the source records; the manuscript specifies the exact
cohorts, sessions, channels, and preprocessing used in each analysis. Access,
use, and redistribution remain governed by the original provider's terms.
Local data paths should be supplied at runtime and must not be committed to this
repository.
