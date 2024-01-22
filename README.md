## ___***FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling***___

### 🔥🔥🔥 The LongerCrafter for longer high-quality video generation are now released!

<div align="center">
<p style="font-weight: bold">
✅ totally <span style="color: red; font-weight: bold">no</span> tuning &nbsp;&nbsp;&nbsp;&nbsp;
✅ less than <span style="color: red; font-weight: bold">20%</span> extra time &nbsp;&nbsp;&nbsp;&nbsp;
✅ support <span style="color: red; font-weight: bold">512</span> frames &nbsp;&nbsp;&nbsp;&nbsp;
</p>

 <a href='https://arxiv.org/abs/2310.15169'><img src='https://img.shields.io/badge/arXiv-2310.15169-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='http://haonanqiu.com/projects/FreeNoise.html'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/MoonQiu/LongerCrafter) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 [![Replicate](https://replicate.com/cjwbw/longercrafter/badge)](https://replicate.com/cjwbw/longercrafter)


_**[Haonan Qiu](http://haonanqiu.com/), [Menghan Xia*](https://menghanxia.github.io), [Yong Zhang](https://yzhang2016.github.io), [Yingqing He](https://github.com/YingqingHe), 
<br>
[Xintao Wang](https://xinntao.github.io), [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ), and [Ziwei Liu*](https://liuziwei7.github.io/)**_
<br><br>
(* corresponding author)

From Tencent AI Lab and Nanyang Technological University.

<img src=assets/t2v/hd01.gif>
<p>Input: "A chihuahua in astronaut suit floating in space, cinematic lighting, glow effect"; 
<br>
Resolution: 1024 x 576; Frames: 64.</p>
<img src=assets/t2v/hd02.gif>
<p>Input: "Campfire at night in a snowy forest with starry sky in the background"; 
<br>
Resolution: 1024 x 576; Frames: 64.</p>
</div>
 
## 🔆 Introduction


🤗🤗🤗 LongerCrafter (FreeNoise) is a tuning-free and time-efficient paradigm for longer video generation based on pretrained video diffusion models.

### 1. Longer Single-Prompt Text-to-video Generation

<div align="center">
<img src=assets/t2v/sp512.gif>
<p>Longer single-prompt results. Resolution: 256 x 256; Frames: 512. (Compressed)</p>
</div>

### 2. Longer Multi-Prompt Text-to-video Generation

<div align="center">
<img src=assets/t2v/mp256.gif>
<p>Longer multi-prompt results. Resolution: 256 x 256; Frames: 256. (Compressed)</p>
</div>

## 📝 Changelog
- __[2023.10.24]__: 🔥🔥 Release the LongerCrafter (FreeNoise), longer video generation!
- __[2023.10.25]__: 🔥🔥 Release the 256x256 model and support multi-prompt generation!
<br>


## 🧰 Models

|Model|Resolution|Checkpoint|Description
|:---------|:---------|:--------|:--------|
<!-- |VideoCrafter (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter)|Support 128 frames on NVIDIA A100 (40GB) -->
|VideoCrafter (Text2Video)|576x1024|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024-v1.0/blob/main/model.ckpt)|Support 64 frames on NVIDIA A100 (40GB)
|VideoCrafter (Text2Video)|256x256|[Hugging Face](https://huggingface.co/VideoCrafter)|Support 512 frames on NVIDIA A100 (40GB)

(Reduce the number of frames when you have smaller GPUs, e.g. 256x256 resolutions with 64 frames.)

## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n freenoise python=3.8.5
conda activate freenoise
pip install -r requirements.txt
```


## 💫 Inference 
### 1. Longer Text-to-Video

<!-- 1) Download pretrained T2V models via [Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-512-v1/blob/main/model.ckpt), and put the `model.ckpt` in `checkpoints/base_512_v1/model.ckpt`.
2) Input the following commands in terminal.
```bash
  sh scripts/run_text2video_freenoise_512.sh
``` -->

1) Download pretrained T2V models via [Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024-v1.0/blob/main/model.ckpt), and put the `model.ckpt` in `checkpoints/base_1024_v1/model.ckpt`.
2) Input the following commands in terminal.
```bash
  sh scripts/run_text2video_freenoise_1024.sh
```

### 2. Longer Multi-Prompt Text-to-Video

1) Download pretrained T2V models via [Hugging Face](https://huggingface.co/VideoCrafter), and put the `model.ckpt` in `checkpoints/base_256_v1/model.ckpt`.
2) Input the following commands in terminal.
```bash
  sh scripts/run_text2video_freenoise_mp_256.sh
```


## 🧲 Support For Other Models

FreeNoise is supposed to work on other similar frameworks. An easy way to test compatibility is by shuffling the noise to see whether a new similar video can be generated (set eta to 0). If your have any questions about applying FreeNoise to other frameworks, feel free to contact [Haonan Qiu](http://haonanqiu.com/).

Current official implementation: [FreeNoise-VideoCrafter](https://github.com/AILab-CVC/FreeNoise), [FreeNoise-AnimateDiff](https://github.com/arthur-qiu/FreeNoise-AnimateDiff), [FreeNoise-LaVie](https://github.com/arthur-qiu/FreeNoise-LaVie) 


## 👨‍👩‍👧‍👦 Crafter Family
[VideoCrafter](https://github.com/AILab-CVC/VideoCrafter): Framework for high-quality video generation.

[ScaleCrafter](https://github.com/YingqingHe/ScaleCrafter): Tuning-free method for high-resolution image/video generation.

[TaleCrafter](https://github.com/AILab-CVC/TaleCrafter): An interactive story visualization tool that supports multiple characters.  


## 😉 Citation
```bib
@misc{qiu2023freenoise,
      title={FreeNoise: Tuning-Free Longer Video Diffusion Via Noise Rescheduling}, 
      author={Haonan Qiu and Menghan Xia and Yong Zhang and Yingqing He and Xintao Wang and Ying Shan and Ziwei Liu},
      year={2023},
      eprint={2310.15169},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****

