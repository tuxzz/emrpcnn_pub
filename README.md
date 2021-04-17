# E-MRP-CNN
This is the original code used in Evolving Multi-Resolution Pooling CNN for Monaural Singing Voice Separation.

You can download pretrained models from https://tuxzz.org/emrpcnn-ckpt/

## Software requirements
* Python 3.7+ 64bit
* CUDA 10.0+
* cuDNN 7.6+
* PIP package: `numpy` `scipy` `matplotlib` `numba` `tensorflow-gpu` `librosa` `sndfile` `mir_eval` `soundfile` `requests` `flask` `frozendict` `psutil` `museval` `filelock`
* Tensorflow 2.0 is recommended to reproduce the paper

## Hardware requirements
* GPU: NVIDIA Pascal and above, with 11GB+ GPU Memory
* SSD: 128GB+ free space for dataset cache
* MEM: 48GB+ free space

## Dataset preparation
* For DSD100 and MIR-1K, no extra processing is needed.
* For MUSDB18, you need decode `.mp4` files into `.wav` files use the `musdb` tool.

## Steps to Evolve
### Server side
1. cd to `srv`
2. Rename the correct config template file to `config.py`
3. If needed, modify the `listen_addr` and `listen_port` in `config.py`.
4. Run `python main_srv.py`
5. Run clients on GPU machine.
6. Your checkpoints will be saved in directory named like `v1_nsga2_mus2_0.0`.

### Client side
1. cd to `cli`
2. Modify the `srv_url` and dataset path in `config.py`
3. Run `python main.py -lockpath=./gpu.lock`

## Steps to Extract Evolution Result
1. cd to `paper_tex`
2. Modify `plot_nsga2_stat.py` line 100 or `plot_g1_stat.py` line 32 to 37 to choose which generation to print.
3. Run `python plot_nsga2_stat.py` or `python plot_nsga2_stat.py` with argument `-input=<evolve checkpoint path> -dataset=<mus2/dsd2/mir2> -score=<train/test>`
4. Copy the gene you need from terminal output, for example `4182591019167972528534244115322478782824676` is a gene, which is used for seed population.

## Steps to Train
### MIR-1K or DSD100
1. cd to `final_eval`
2. Modify `config.py`, set correct cache path and dataset path.
3. Run `python dsd2_mkcache.py` or `python mir2_mkcache.py`
4. Run `python dsd2_train.py` or `python mir2_train.py` with argument `-ver=v1 -gene=<gene number>`.
5. The checkpoints are saved in path like `ckpt/<mir2/dsd2>_v1_<gene number>`.

### MUSDB18
1. cd to `final_eval`
2. Modify `config.py`, set correct dataset path.
3. Modify `mus2f_train.py` line 133:138, set correct `cache_meta_path` and `cache_path`.
4. Run `python mus2f_train.py -ver=v1fm -gene=<gene number>`
5. The checkpoints are saved in path like `ckpt/mus2f_v1fm_<gene number>`.

## Steps to Evaluate
### MIR-1K or DSD100
1. cd to `final_eval`
2. Modify `config.py`, set correct dataset path.
3. Run `python dsd2_eval.py` or `python mir2_eval.py` with argument `-ver=v1f -gene=<gene number> -step=<checkpoint step>`
4. The result is saved in `eval_output`.

### MUSDB18
1. cd to `final_eval`
2. Modify `config.py`, set correct dataset path.
3. Run `python mus2f_museval.py -ver=v1fm -gene=<gene number> -step=<checkpoint step>`
4. The result is saved in `eval_output_mus2f` and `eval_output_mus2f_bundle`, the splitted audio is saved in `sound_output_mus2f`.

## Steps to Split Any Audio
### Using MUSDB18 Checkpoints
1. cd to `final_eval`
2. Run `python split_song_f.py  -dataset=mus2 -ver=v1fm -gene=<gene number> -step=<checkpoint step> -input=<wav path>`
3. The result is saved in `split_out_f`.
