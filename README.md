\# Multi-class Road Extraction from Slovenian VHR Orthophotos



Code repository for the paper:



> Stojanović, L., Fetai, B., Lisec, A. (2026). Multi-class road extraction from

> high-resolution orthophotos with deep learning: a landscape-stratified evaluation

> for Slovenia. \*ISPRS Journal of Photogrammetry and Remote Sensing\*.



\## Overview



This repository contains the complete pipeline for extracting functionally

differentiated road networks from 0.25 m Slovenian national orthophotos using

semantic segmentation. Four architectures are compared — U-Net/ResNet50,

D-LinkNet, SegFormer-B2, and DeepLabV3+ — across three road classes (major,

local, minor) and seven landscape types.



\*\*Key results:\*\*

\- Best model: SegFormer-B2, Road Macro F1 = 0.591 (3-class), 0.605 (2-class)

\- 20 municipalities, spatially separated train/val/test split

\- 261,234 training tiles at 512×512 px, 0.25 m GSD

\- Total compute: \~144 GPU-hours on NVIDIA H100 NVL



\## Repository Structure

├── notebooks/          # Data preparation (run in order 01–06)

├── training/           # Training scripts (run from CLI, not Jupyter)

├── evaluation/         # Evaluation and connectivity metrics

├── analysis/           # Statistical analysis and paper artifacts

├── visualization/      # Figure generation

├── data/splits/        # Train/val/test municipality lists

├── config.yaml         # Central configuration

└── environment.yml     # Conda environment



\## Installation



```bash

conda env create -f environment.yml

conda activate roads\_env

```



\## Data



Orthophotos are available from GURS (Geodetic Administration of Slovenia):

https://www.e-prostor.gov.si



The official road network (roads\_102109.gpkg) and municipality boundaries

(municipalities\_102109.gpkg) are also available from GURS.



Before running any script, set environment variables:



\*\*Windows:\*\*

```bash

set ROOT\_DIR=D:\\your\\path\\to\\road\_extraction\_slovenia

set ORTHO\_DIR=\\\\your\\nas\\path\\to\\DOF025

```



\*\*Linux:\*\*

```bash

export ROOT\_DIR=/your/path/to/road\_extraction\_slovenia

export ORTHO\_DIR=/your/path/to/DOF025

```



\## Pre-trained Models



Model weights (\~500 MB) are available on Zenodo:

\[DOI link — to be added upon acceptance]



Download and place in: `$ROOT\_DIR/models/`



Expected structure:

models/

├── unet\_resnet50/unet\_resnet50\_best.pth

├── dlinknet/dlinknet\_best.pth

├── segformer/segformer\_best.pth

├── deeplabv3plus/deeplabv3plus\_best.pth

└── 2class/

├── unet\_resnet50/unet\_resnet50\_best.pth

├── dlinknet/dlinknet\_best.pth

├── segformer/segformer\_best.pth

└── deeplabv3plus/deeplabv3plus\_best.pt



\## Reproduction Pipeline



\### Step 1 — Data preparation (Jupyter notebooks)

Run notebooks in order from the `notebooks/` folder:

01\_explore\_municipality\_data.ipynb

02\_explore\_road\_data.ipynb

03\_prepare\_road\_labels.ipynb

04\_select\_orthophotos.ipynb

05\_remap\_masks\_2class.ipynb

06\_generate\_tiles.ipynb



\### Step 2 — Training (run from Anaconda Prompt, not Jupyter)

```bash

python training/train\_3class.py

python training/train\_2class.py

```



\### Step 3 — Evaluation

```bash

python evaluation/test\_3class.py

python evaluation/test\_2class.py

python evaluation/table9\_all\_models\_connectivity.py

```



\### Step 4 — Statistical analysis

```bash

python analysis/artifact1\_per\_tile\_counts.py

python analysis/artifact1\_analysis.py

python analysis/artifact3\_failure\_cases.py

python analysis/artifact4\_geotiff\_export.py

```



\### Step 5 — Figures

```bash

python visualization/generate\_article\_data.py

python visualization/generate\_visual\_maps.py

python analysis/figure5\_error\_analysis.py

```



\## Municipality Splits



| Split | Municipalities | Landscape types |

|---|---|---|

| Train (12) | Ljubljana, Maribor, Koper, Celje, Ptuj, Murska Sobota, Krško, Postojna, Kočevje, Bovec, Ajdovščina, Slovenska Bistrica | Urban, coastal, agricultural, karst, forest, alpine, suburban |

| Val (3) | Kranj, Novo mesto, Tolmin | Suburban-alpine, suburban, alpine |

| Test (5) | Nova Gorica, Domžale, Bohinj, Piran, Lendava | Urban, suburban, alpine, coastal, agricultural |



\## Road Class Scheme



| Class | Name | KAT\_CES codes | Min buffer |

|---|---|---|---|

| 1 | Major roads | AC, HC, G1, G2, R1 | 3.0 m |

| 2 | Local roads | GZ, R2, R3, RT, LC, LZ | 2.0 m |

| 3 | Minor roads | LG, LK, JP, KD, KJ, NK, PP | 1.5 m |



\## Citation



```bibtex

@article{stojanovic2025road,

&#x20; title   = {Multi-class road extraction from high-resolution orthophotos

&#x20;            with deep learning: a landscape-stratified evaluation for Slovenia},

&#x20; author  = {Stojanović, Luka and Fetai, Bujar and Lisec, Anka},

&#x20; journal = {ISPRS Journal of Photogrammetry and Remote Sensing},

&#x20; year    = {2025}

}

```



\## License



This code is released under the MIT License. See LICENSE for details.



The orthophoto data is subject to GURS licensing terms and is not included

in this repository.

