<img src="images/Provost_Research_Computing.png" alt="drawing" width="900"/>

# Research Computing Training

## Presenter

Ghanghoon "Will" Paik

HPC Performance Engineer

[Research Computing](https://rc.northeastern.edu/research-computing-team/)

## Running PyTorch Training Jobs on Explorer

Welcome to the [Research Computing Fall 2026 Training Series](https://rc.northeastern.edu/research-computing-fall-training/)!

During this session, we will set up a working PyTorch environment on Explorer, using either conda or a shared container, and turn a Python training script into a batch job. Along the way we will look at how to pick a GPU, why the default PyTorch install does not work on Explorer right now, and how to fit your training into the time limit of the GPU partition.

Today's Agenda:

[1. Choosing a GPU on Explorer](#1-choosing-a-gpu-on-explorer)  
[2. Matching PyTorch to the GPU](#2-matching-pytorch-to-the-gpu)  
[3. Option A: Installing PyTorch with Conda](#3-option-a-installing-pytorch-with-conda)  
[4. Option B: Using the Shared PyTorch Container](#4-option-b-using-the-shared-pytorch-container)  
[5. Interactive Development with Jupyter on OOD](#5-interactive-development-with-jupyter-on-ood)  
[6. From Notebook to Batch Job](#6-from-notebook-to-batch-job)  
[7. Summary and Resources](#7-summary-and-resources)  

Sample codes used in this session are available at **[Running-PyTorch-Training-Jobs-2026](https://github.com/northeastern-rc-training/Running-PyTorch-Training-Jobs-2026)**

Includes: `train.py` (ResNet-18 on CIFAR-10), `download_data.py`, batch scripts for the download and for training with conda or the container, and the container definition file.

---

## 1. Choosing a GPU on Explorer

### 1.1 GPUs in the `gpu` Partition

| GPU | `--gres` name | Memory | Architecture (compute capability) | Native BF16 | Availability |
|---|---|---|---|---|---|
| T4 | `t4` | 16 GB | Turing (7.5) | No | Few |
| V100 PCIe | `v100-pcie` | 32 GB | Volta (7.0) | No | Many |
| V100 SXM2 | `v100-sxm2` | 32 GB | Volta (7.0) | No | Many |
| A100 | `a100` | 40/80 GB | Ampere (8.0) | Yes | Few |
| H200 | `h200` | 141 GB | Hopper (9.0) | Yes | Few |

Request a specific type with `--gres`:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
```

Leave out the type (`--gres=gpu:1`) and you get whichever GPU becomes free first.

To see which GPUs are in the partition and how many nodes are idle:

```bash
sinfo -p gpu -o "%25G %10t %5D"
```

Sample output:

```
GRES                      STATE      NODES
gpu:v100-pcie:2(S:0-7)    down*      1    
gpu:v100-sxm2:4(S:0-1)    comp       4    
gpu:h200:8                comp       1    
gpu:v100-pcie:2(S:0-1)    mix        3    
gpu:v100-sxm2:4(S:0-1)    mix        5    
gpu:v100-sxm2:3(S:0-1)    mix        1    
gpu:t4:4(S:0-1)           mix        1    
gpu:a100:3                mix        1    
gpu:a100:4                mix        2    
gpu:h200:8                mix        3    
gpu:v100-sxm2:4(S:0-1)    alloc      2    
gpu:v100-sxm2:3           alloc      1  
```

### 1.2 What Uses GPU Memory

During training, the GPU holds:

- **Model weights**
- **Gradients**: one value per weight
- **Optimizer state**: SGD with momentum keeps one extra copy of the weights, Adam keeps two
- **Activations**: values saved during the forward pass for the backward pass. These grow with the batch size.

A rule of thumb for FP32 training with Adam is 16 bytes per parameter before activations (4 for the weight, 4 for the gradient, 8 for Adam). A model with 1 billion parameters needs about 16 GB just for that. It does not fit on a T4, and on a V100 half of the memory is gone before the first batch.

When the GPU runs out of memory, PyTorch stops with an error that starts like this:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```

What to try, in order:

1. Reduce the batch size.
2. Move to a GPU with more memory (A100 40/80 GB or H200 141 GB).
3. Use memory saving techniques such as mixed precision. These are covered in the Optimizing GPU usage session.

### 1.3 Which One Should I Pick?

- **Start with a V100.** There are many more of them than A100s and H200s, so V100 jobs tend to start sooner.
- **T4** is fine for quick tests and for running small models.
- **Move to an A100 or H200** when:
  - your job runs out of memory on a V100 even with a smaller batch,
  - you need BF16 (many recent models are trained and released in BF16), or
  - your training does not fit in 8 hours on a V100.

A bigger GPU is not always faster. If your model is small, or the GPU spends its time waiting for data, an H200 will not help much. The Optimizing GPU usage session shows how to check how busy your GPU actually is.

### 1.4 Partition Limits

| Partition | Max time | GPUs per job | Access |
|---|---|---|---|
| `gpu` | 8 hours | 1 | All users |
| `gpu-interactive` | 2 hours | 1 | All users |
| `multigpu` | 24 hours | 8 | Application and approval required |

This session uses the `gpu` partition only. If your training needs more than 8 hours, see [Section 6.4](#64-fitting-in-8-hours).

---

## 2. Matching PyTorch to the GPU

Three pieces have to agree before PyTorch can use a GPU:

| Piece | What it is | Who controls it | How to check |
|---|---|---|---|
| NVIDIA driver | Software on the GPU node that talks to the GPU | Research Computing | `nvidia-smi` |
| PyTorch CUDA build | The CUDA version PyTorch was compiled with | You, when you install PyTorch | `torch.version.cuda` |
| GPU architecture | The generation of the GPU | The node your job lands on | `torch.cuda.get_device_capability()` |

Two rules:
Choose the gpu partition and a GPU type.
1. **The driver must support the CUDA version of your PyTorch build.** Drivers on Explorer support CUDA 12.x. A CUDA 12.x build of PyTorch works. A CUDA 13.x build does not.
2. **The build must include code for your GPU.** `torch.cuda.get_arch_list()` shows which architectures are in the build. A V100 (compute capability 7.0) needs `sm_70` in that list.

The `CUDA Version` in the top right corner of `nvidia-smi` is the highest CUDA version the driver supports. It does not mean a CUDA toolkit is installed.

| GPU | Driver supports up to | Compute capability |
|---|---|---|
| V100 (PCIe, SXM2) | CUDA 12.3 | 7.0 (`sm_70`) |
| T4 | CUDA 12.8 | 7.5 (`sm_75`) |
| A100 | CUDA 12.8 | 8.0 (`sm_80`) |
| H200 | CUDA 12.8 | 9.0 (`sm_90`) |

> Driver versions as of October 2026. Check `nvidia-smi` on your node for the current value.

### 2.1 The Default Install Does Not Work on Explorer

The command on pytorch.org (`pip install torch`) currently installs a CUDA 13.0 build. On an Explorer V100 node:

```
>>> import torch
>>> torch.__version__
'2.14.1+cu130'
>>> torch.cuda.is_available()
.../torch/cuda/__init__.py:228: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12030). ...
False
```

`found version 12030` is how CUDA writes 12.3 (12 x 1000 + 3 x 10).

Watch out: `is_available()` only prints a warning and returns `False`. Many training scripts pick the device like this:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

With the wrong build, that script does not crash. It trains on the CPU, very slowly, while holding a GPU. The sample `train.py` stops with an error instead.

This build also leaves out `sm_70`, so it would not run on a V100 even with a newer driver:

```
>>> torch.cuda.get_arch_list()
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

### 2.2 What Works: PyTorch 2.14.1 with CUDA 12.6

```
>>> torch.__version__
'2.14.1+cu126'
>>> torch.version.cuda
'12.6'
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'Tesla V100-SXM2-32GB'
>>> torch.cuda.get_arch_list()
['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
```

The CUDA 12.6 build runs on the CUDA 12.3 driver of the V100 nodes, because a driver for one CUDA 12 release can run programs built with a newer CUDA 12 release (CUDA minor version compatibility). The build also includes every GPU architecture on Explorer (tested on V100).

> PyTorch 2.14 is the last release with CUDA 12 builds. Prebuilt PyTorch 2.15 and later need a driver that supports CUDA 13, and they no longer include V100 (`sm_70`). On Explorer, stay on PyTorch 2.14 for now.

### 2.3 Do I Need `module load cuda`?

Not to run PyTorch. When you install PyTorch with pip, the CUDA libraries it needs (CUDA runtime, cuDNN, NCCL) are installed along with it. Only the driver comes from the node.

You need a CUDA toolkit module only when you compile CUDA code yourself, for example when you build a PyTorch extension from source.

---

## 3. Option A: Installing PyTorch with Conda

Conda creates the Python environment, and pip installs PyTorch into it. PyTorch no longer publishes its own conda packages, so pip is the supported route.

### 3.1 Create the Environment

Run this in an interactive job, not on the login node:

```bash
srun --partition=short --cpus-per-task=4 --mem=8G --pty bash

module load miniconda3/25.9.1
conda create -n pytorch_env python=3.12 -y
source activate pytorch_env
pip install torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126
```

`--index-url` tells pip to look only at PyTorch's CUDA 12.6 package index. Without it, pip installs the CUDA 13.0 build from PyPI. Do not use `--extra-index-url` here either: pip then looks at both places and takes the highest version it finds, which can be the CUDA 13 build.

> Note: The environment takes \~6.5 GB. Keep it in your home or a project directory, not on scratch. See Section 3.3 of the [Software Installation training](https://github.com/northeastern-rc-training/Intro-to-Software-Installation-2026) for where to put conda environments. pip also keeps a cache of downloaded packages in `~/.cache/pip`. Clear it with `pip cache purge` if your home quota is tight.

### 3.2 Check It on a GPU Node

The login node and CPU nodes have no GPU, so this check has to run in a GPU job:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=15:00 --pty bash

module load miniconda3/25.9.1
source activate pytorch_env
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected output:

```
2.14.1+cu126 12.6 True
```

### 3.3 Adding More Packages

Install other packages with pip as usual. One thing to watch: if a package asks for a newer PyTorch, pip replaces yours with the default build from PyPI, which is CUDA 13 and does not work on Explorer.

Two habits prevent this:

- Add `torch==2.14.1` to the same command. If a package needs a newer PyTorch, pip then stops with an error instead of replacing it.

  ```bash
  pip install transformers matplotlib torch==2.14.1
  ```

- After installing packages, run the check in Section 3.2 again. `torch.version.cuda` should still say `12.6`.

---

## 4. Option B: Using the Shared PyTorch Container

Research Computing provides an Apptainer image with PyTorch already installed. There is nothing to install, and nothing goes into your home quota.

Location: `/shared/container_repository/explorer/pytorch/pytorch_training.sif`

What is inside:

- Python 3.12, PyTorch 2.14.1 (CUDA 12.6), torchvision
- numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, tqdm
- h5py, pyarrow, zarr
- transformers, accelerate, datasets, safetensors, optuna
- pillow, opencv-python-headless, tensorboard, ipykernel

The exact versions are recorded inside the image:

```bash
apptainer exec /shared/container_repository/explorer/pytorch/pytorch_training.sif cat /opt/pip-freeze.txt
```

### 4.1 Run Python Inside the Container

```bash
apptainer exec --nv -B /scratch /shared/container_repository/explorer/pytorch/pytorch_training.sif \
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- `exec` runs the command you give it inside the container.
- `--nv` makes the GPU and the node's NVIDIA driver visible inside the container. Without it, `torch.cuda.is_available()` returns `False`.
- `-B` binds directories into the container. Only your home directory is available by default, so add `/scratch` or `/projects` if your code or data live there.

> The container uses the driver of the node through `--nv`, so the rules from Section 2 still apply. A container with a CUDA 13 build of PyTorch would fail on Explorer in exactly the same way.

### 4.2 Need Another Package?

Do not `pip install` inside the container. The image is read-only, and `pip install --user` puts packages in `~/.local` in your home directory. The image ignores `~/.local` on purpose, so those packages will not show up inside it, but they will show up in your other Python environments and can break them.

Instead, copy the definition file from the sample repository (`container/pytorch_training.def`), add your packages, and build your own image. Section 7 of the [Software Installation training](https://github.com/northeastern-rc-training/Intro-to-Software-Installation-2026) shows how to build an image on your own machine and copy it to the cluster.

### 4.3 Conda or Container?

| | Conda + pip | Shared container |
|---|---|---|
| Setup | You install it yourself | Nothing to install |
| Adding packages | Easy | Rebuild the image (needs a machine with root) |
| Home quota | \~6.5 GB, tens of thousands of files | None, the image lives in a shared location |
| Filesystem load | Many small files read at every import | One file |
| Good for | Development, changing packages often | Standard packages, running the same setup in many jobs |

Same `train.py` on the same V100-SXM2 node, 4 CPU cores, 10 epochs:

| Environment | Train time per epoch | Total time | Final test accuracy |
|---|---|---|---|
| Conda | 11.3 s | 131.3 s | 76.93% |
| Container | 11.1 s | 126.6 s | 78.42% |

Train time per epoch is the average of epochs 2 to 10. The first epoch also includes startup.

Things to notice:

- The container adds no meaningful overhead. A few percent of difference between two runs is normal. In an earlier pair of runs, conda was the faster one.
- The accuracy differs slightly because the model starts from random weights in each run.

---

## 5. Interactive Development with Jupyter on OOD

Jupyter Notebook is a good tool to develop and test before you submit a batch job. The JupyterLab Notebook app on OOD can use your own conda environment if `jupyterlab` is installed in it. You can install it with:

```bash
module load miniconda3/25.9.1
source activate pytorch_env
conda install -y jupyterlab
```

![Jupyter Notebook Configuration](images/jupyternotebook.png)

1. On OOD, open the JupyterLab app.
2. Choose the `gpu-interactive` partition and a GPU type.
3. Check **Custom Anaconda Environment (provide name only)** and enter `pytorch_env` in **Name of Custom Conda Environment**.
4. Check the GPU in a notebook cell:

   ```python
   import torch
   print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
   ```

5. Run a short training. Write the path out in full, because `$USER` is not expanded inside `%run`:

   ```python
   %run train.py --epochs 2 --data-dir /scratch/<username>/data
   ```

Long training goes into a batch job, not a notebook. When you are done testing, end the session so the GPU goes back to other users.

---

## 6. From Notebook to Batch Job

### 6.1 Download Data Before the GPU Job

Downloading needs no GPU. Do it once, in a separate CPU job, and point every training job at that copy.

`download_data.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=download-cifar10
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

DATA_DIR=/scratch/$USER/data

module load miniconda3/25.9.1
source activate pytorch_env
python download_data.py --data-dir "$DATA_DIR"
```

Why a separate job:

- **Downloads can be slow.** While preparing this session, CIFAR-10 (170 MB) once took 43 minutes to download inside a GPU job. The V100 sat idle the whole time, and nobody else could use it.
- **One copy for every job.** Relative paths like `./data` create a new copy of the dataset in every directory you run from. Use one fixed path with `--data-dir`.

`train.py` never downloads anything. If the data is not there, it stops right away instead of holding the GPU:

```
ERROR: CIFAR-10 dataset not found in /scratch/<user>/data.
  Please download it first:
  python download_data.py --data-dir /scratch/<user>/data
```

You can submit both jobs at once and let Slurm start training only after the download finishes:

```bash
jobid=$(sbatch --parsable download_data.sh)
sbatch --dependency=afterok:$jobid train_conda.sh
```

`--parsable` makes `sbatch` print only the job ID. `afterok` means "start only if that job finished without errors". The training job waits in the queue without holding a GPU. If the download fails, the training job stays pending with the reason `DependencyNeverSatisfied`. Cancel it with `scancel` and check the download log.

The same idea applies to your own data: model weights from Hugging Face, datasets from the web, or files from another server. Get them in a CPU job or on the transfer node first.

### 6.2 The Batch Script

`train_conda.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=pytorch-training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00
#SBATCH --output=%x-%j.out

DATA_DIR=/scratch/$USER/data

module load miniconda3/25.9.1
source activate pytorch_env

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

python train.py --data-dir "$DATA_DIR" --epochs 10 "$@"
```

| Line | What it does |
|---|---|
| `--partition=gpu` | GPU partition: up to 8 hours, 1 GPU |
| `--gres=gpu:v100-sxm2:1` | One V100-SXM2. Use `--gres=gpu:1` for any type |
| `--cpus-per-task=4` | CPU cores for the job. `train.py` uses all but one of them to load data |
| `--mem=16G` | CPU memory (RAM). This is not GPU memory |
| `--time=30:00` | Time limit. Slurm stops the job when it is reached |
| `--output=%x-%j.out` | Log file named after the job name and job ID. Output and errors both go here |
| `module load`, `source activate` | Activate the environment inside the job, not in `.bashrc` |
| `nvidia-smi` | Records which GPU and driver the job got. Useful when you ask for help |
| `"$@"` | Passes extra arguments to `train.py`: `sbatch train_conda.sh --epochs 3` |

For the container, only the last part changes (`train_container.sh`):

```bash
CONTAINER=/shared/container_repository/explorer/pytorch/pytorch_training.sif

apptainer exec --nv -B /scratch "$CONTAINER" \
    python train.py --data-dir "$DATA_DIR" --epochs 10 "$@"
```

Submit and follow the log:

```bash
sbatch train_conda.sh
squeue -u $USER
tail -f pytorch-training-<jobid>.out
```

If your own script shows nothing in the log for a long time, Python may be holding the output in a buffer. Use `python -u` or `print(..., flush=True)`.

### 6.3 How Many CPUs?

While the GPU trains on one batch, DataLoader workers on the CPU prepare the next ones: read the images, crop and flip them, and turn them into tensors. `train.py` starts one worker per CPU core, minus one core for the main process.

Same job on a V100-SXM2, 3 epochs, changing only `--cpus-per-task`:

| `--cpus-per-task` | DataLoader workers | Train time per epoch | Images per second |
|---|---|---|---|
| 1 | 1 | 17.2 s | 2,900 |
| 4 | 3 | 11.2 s | 4,500 |
| 8 | 7 | 11.1 s | 4,500 |

Averages of epochs 2 and 3.

```bash
for c in 1 4 8; do
    sbatch --cpus-per-task=$c --job-name=cpu$c train_conda.sh --epochs 3 --output-dir ./output_cpu$c
done
```

Things to notice:

- With 1 core, each epoch took about 50% longer than with 4. The GPU was waiting for the CPU to prepare images.
- Going from 4 to 8 cores changed nothing. Once the data arrives fast enough, more cores do not help.
- Unless you ask for more, Slurm gives your job 1 core. A GPU job without `--cpus-per-task` can be much slower than it should be.
- 4 cores per GPU is a good starting point for jobs like this one. Measure your own job the same way. Asking for more cores than you use does not make it faster, and leaves fewer cores for other GPU jobs on the same node.

If you set more DataLoader workers than you have cores, PyTorch prints a warning like this one. It usually means `--cpus-per-task` is missing:

```
UserWarning: This DataLoader will create 3 worker processes in total. Our suggested max number of worker in current system is 1, which is smaller than what this DataLoader is going to create. ...
```

Tuning the data pipeline further is covered in the Optimizing GPU usage session.

### 6.4 Fitting in 8 Hours

Run 2 or 3 epochs first and read the total time at the end of the log. Use the total, not the per-epoch `train time`, because the total also includes evaluation and saving the checkpoint.

Example from the demo: 10 epochs on a V100 with 4 cores took about 130 seconds in total, about 13 seconds per epoch. That is 2 seconds more than the `train time`, for evaluation and the checkpoint. 100 epochs would take about 22 minutes, so `--time=00:30:00` leaves some room.

When a job reaches its time limit, Slurm stops it and the log ends with a line like:

```
slurmstepd: error: *** JOB <jobid> ON <node> CANCELLED AT <time> DUE TO TIME LIMIT ***
```

`train.py` saves `output/checkpoint.pt` after every epoch. If the job is stopped, submit the same script again and it continues from the last finished epoch:

```
Resumed from output/checkpoint.pt, starting at epoch 7
```

To start over, delete the output directory. Give separate runs separate output directories (`--output-dir`), or they will overwrite and resume from each other's checkpoints.

Checkpointing in more depth is covered in the Checkpoint Restart on HPC session.

---

## 7. Summary and Resources

### Key Takeaways

1. **Install PyTorch 2.14.1 with CUDA 12.6 on Explorer.** The default `pip install torch` gives a CUDA 13 build that does not work with the current drivers.
2. **Check on a GPU node.** `torch.version.cuda` should be `12.6` and `torch.cuda.is_available()` should be `True`.
3. **Conda or container.** Conda when you change packages often, the shared container when the standard packages are enough. Performance is the same.
4. **Download data in a separate CPU job.** Never let a GPU wait for a download.
5. **Ask for CPU cores, not only a GPU.** With 1 core, the demo took about 50% longer per epoch. 4 cores were enough.
6. **Start with a V100.** Move to an A100 or H200 when memory, BF16, or time requires it.
7. **Plan for 8 hours.** Measure a short run, then checkpoint and resubmit if you need more time.

### Quick Reference

```bash
# Conda environment (in an interactive job)
module load miniconda3/25.9.1
conda create -n pytorch_env python=3.12 -y
source activate pytorch_env
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126

# Check on a GPU node
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

# Shared container
apptainer exec --nv -B /scratch /shared/container_repository/explorer/pytorch/pytorch_training.sif \
    python train.py --data-dir /scratch/$USER/data

# Download data in a CPU job, then train in a GPU job
jobid=$(sbatch --parsable download_data.sh)
sbatch --dependency=afterok:$jobid train_conda.sh
squeue -u $USER
```

### Sample Code Repository

**[Running-PyTorch-Training-Jobs-2026](https://github.com/northeastern-rc-training/Running-PyTorch-Training-Jobs-2026)**

Includes: `train.py`, `download_data.py`, `download_data.sh`, `train_conda.sh`, `train_container.sh`, and `pytorch_training.def`

### Getting Help

Email the Research Computing team at [rchelp@northeastern.edu](mailto:rchelp@northeastern.edu).

Come to [office hours](https://rc.northeastern.edu/getting-help/) hosted on Zoom.

Or [book a consultation](https://rc.northeastern.edu/getting-help/) with an RC team member.

Review our [Documentation](https://rc-docs.northeastern.edu/en/latest/index.html).

Thank you!

---

_For questions or support, contact the Research Computing team at [rchelp@northeastern.edu](mailto:rchelp@northeastern.edu)_
