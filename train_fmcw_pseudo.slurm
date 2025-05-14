#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=todiit00@hs-esslingen.de
#SBATCH --job-name=fmcw_pseudo
#SBATCH --output=slurm_%j.out

# Lade CUDA-Toolchain
module load devel/cuda/12.4

# Wechsle ins Scratch-Verzeichnis (schneller Speicher)
cd $TMPDIR

# Kopiere Code + Config # Daten
rsync -avz $HOME/paconv/obj_cls/ ./obj_cls/
rsync -avz $HOME/paconv/obj_cls/config/fmcw_dummy_train.yaml ./obj_cls/config/
rsync -avz --inplace $HOME/paconv/requirements.txt .
rsync -avz $HOME/paconv/data/fmcw_test/ ./data/fmcw_test/

# Virtuelle Umgebung vorbereiten
python3 -m venv paconv-env
source paconv-env/bin/activate

# Installiere Abhängigkeiten
pip install --upgrade pip
pip install -r requirements.txt

# Starte das Pseudo-Training
time python3 obj_cls/main.py --config obj_cls/config/fmcw_dummy_train.yaml

# Kopiere Ergebnisse zurück ins HOME
mkdir -p $HOME/paconv/results
rsync -avz checkpoints/ $HOME/paconv/results/checkpoints_fmcw_$SLURM_JOB_ID/
