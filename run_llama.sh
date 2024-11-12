#!/bin/bash
#SBATCH -J llama_3.2_vision            #Job name(--job-name)
#SBATCH -o %j.out_          #Name of stdout output file(--output)
#SBATCH -e %j.err_          #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name ; available options "shared,medium,large or gpu" 
#SBATCH -N 2                       # no of Nodes
#SBATCH -n 8                       # no of processes or tasks
#SBATCH --gres=gpu:2               # request gpu card: it should be either 1 or 2
#SBATCH --cpus-per-task=4          # no of threads per process or task
#SBATCH -t 02:30:00         # specifies walltime(--time maximum duration)of run
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
#SBATCH --mail-user=theguysta56@gmail.com        # user's email ID where job status info will be sent


module load compiler/cuda/11.7  
source ~/miniconda3/bin/activate embodied 

echo "Job started on $(hostname) at $(date)"
nvidia-smi

#python /scratch/22ch10090/visionlanguage/inria/llama_model.py
python /scratch/22ch10090/visionlanguage/inria/run_llama.py

