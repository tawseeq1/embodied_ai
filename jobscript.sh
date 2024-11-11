#!/bin/bash
#SBATCH -J inria1            #Job name(--job-name)
#SBATCH -o %j.err_          #Name of stdout output file(--output)
#SBATCH -e %j.out_          #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name ; available options "shared,medium,large or gpu" 
#SBATCH -N 2                       # no of Nodes
#SBATCH -n 8                       # no of processes or tasks
#SBATCH --gres=gpu:1               # request gpu card: it should be either 1 or 2
#SBATCH --cpus-per-task=4          # no of threads per process or task
#SBATCH -t 00:57:00         # specifies walltime(--time maximum duration)of run
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
#SBATCH --mail-user=theguysta56@gmail.com        # user's email ID where job status info will be sent


module load compiler/cuda/11.7
source /home/22ch10090/miniconda3/etc/profile.d/conda.sh && conda activate embodied
python /scratch/22ch10090/visionlanguage/inria/models.py


