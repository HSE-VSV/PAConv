#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=todiit00@hs-esslingen.de
#SBATCH --job-name=paconv_train
#SBATCH --output=slurm_%j.out

# CUDA-Toolchain laden
module load devel/cuda/12.4

# Wechsle ins temporäre Verzeichnis
cd $TMPDIR

# Kopiere Projekt + Daten aus dem HOME-Verzeichnis
rsync -avz $HOME/paconv/obj_cls/ ./obj_cls/
rsync -avz $HOME/paconv/data/ ./data/
rsync -avz $HOME/paconv/requirements.txt .

# Erstelle virtuelle Umgebung
python3 -m venv paconv-env
source paconv-env/bin/activate

# Installiere Abhängigkeiten (inkl. Ninja)
pip install --upgrade pip
pip install -r requirements.txt

# Starte Training
time python3 obj_cls/main.py --config obj_cls/config/dgcnn_paconv_train.yaml

# Optional: Kopiere Ergebnisse zurück ins HOME-Verzeichnis
mkdir -p $HOME/paconv/results
rsync -avz obj_cls/checkpoints/ $HOME/paconv/results/checkpoints_$SLURM_JOB_ID/
