#!/usr/bin/env bash
#SBATCH --job-name=moss-der-audit
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/work/qt28/moss/analysis/20260915-label-overlap/der-%j.out
#SBATCH --error=/work/qt28/moss/analysis/20260915-label-overlap/der-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2
/work/qt28/moss/dkucc/evaluation/metrics-env/bin/python \
  /work/qt28/moss/analysis/20260915-label-overlap/der_supplement.py
