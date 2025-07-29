#!/bin/bash
#
#SBATCH --job-name=concept_collection
#SBATCH --output=/storage/ukp/work/kaliu/res.txt
#SBATCH --mail-user=kai.li@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1


source /storage/ukp/work/kaliu/miniconda3/bin/activate thesis

python /storage/ukp/work/kaliu/concept_collection_pipeline/concept_filtering.py
