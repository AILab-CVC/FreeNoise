# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import os
import sys
import argparse
import random
from omegaconf import OmegaConf
from einops import rearrange, repeat
import torch
import torchvision
from pytorch_lightning import seed_everything
from cog import BasePredictor, Input, Path

sys.path.insert(0, "scripts/evaluation")
from funcs import (
    batch_ddim_sampling_freenoise,
    load_model_checkpoint,
    load_image_batch,
    get_filelist,
)
from utils.utils import instantiate_from_config


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        ckpt_path_1024 = "checkpoints/base_1024_v1/model.ckpt"
        config_1024 = "configs/inference_t2v_1024_v1.0_freenoise.yaml"
        ckpt_path_256 = "checkpoints/base_256_v1/model.pth"
        config_256 = "configs/inference_t2v_tconv256_v1.0_freenoise.yaml"

        config_1024 = OmegaConf.load(config_1024)
        model_config_1024 = config_1024.pop("model", OmegaConf.create())
        self.model_1024 = instantiate_from_config(model_config_1024)
        self.model_1024 = self.model_1024.cuda()
        self.model_1024 = load_model_checkpoint(self.model_1024, ckpt_path_1024)
        self.model_1024.eval()

        config_256 = OmegaConf.load(config_256)
        model_config_256 = config_256.pop("model", OmegaConf.create())
        self.model_256 = instantiate_from_config(model_config_256)
        self.model_256 = self.model_256.cuda()
        self.model_256 = load_model_checkpoint(self.model_256, ckpt_path_256)
        self.model_256.eval()

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for video generation.",
            default="A chihuahua in astronaut suit floating in space, cinematic lighting, glow effect.",
        ),
        output_size: str = Input(
            description="Choose the size of the output video.",
            choices=["576x1024", "256x256"],
            default="576x1024",
        ),
        num_frames: int = Input(
            description="Number for frames to generate.", default=32
        ),
        ddim_steps: int = Input(description="Number of denoising steps.", default=50),
        unconditional_guidance_scale: float = Input(
            description="Classifier-free guidance scale.", default=12.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        save_fps: int = Input(
            description="Frame per second for the generated video.", default=10
        ),
        window_size: int = Input(description="Window size.", default=16),
        window_stride: int = Input(description="Window stride.", default=4),
    ) -> Path:

        width = 1024 if output_size == "576x1024" else 256
        height = 576 if output_size == "576x1024" else 256
        model = self.model_1024 if output_size == "576x1024" else self.model_256

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        seed_everything(seed)

        args = argparse.Namespace(
            mode="base",
            savefps=save_fps,
            n_samples=1,
            ddim_steps=ddim_steps,
            ddim_eta=0.0,
            bs=1,
            height=height,
            width=width,
            frames=num_frames,
            fps=24,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_guidance_scale_temporal=None,
            cond_input=None,
            window_size=window_size,
            window_stride=window_stride,
        )

        ## latent noise shape
        h, w = args.height // 8, args.width // 8
        frames = model.temporal_length if args.frames < 0 else args.frames
        channels = model.channels

        x_T_total = torch.randn(
            [args.n_samples, 1, channels, frames, h, w], device=model.device
        ).repeat(1, args.bs, 1, 1, 1, 1)
        for frame_index in range(args.window_size, args.frames, args.window_stride):
            list_index = list(
                range(
                    frame_index - args.window_size,
                    frame_index + args.window_stride - args.window_size,
                )
            )
            random.shuffle(list_index)
            x_T_total[
                :, :, :, frame_index : frame_index + args.window_stride
            ] = x_T_total[:, :, :, list_index]

        batch_size = 1
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps] * batch_size).to(model.device).long()
        prompts = [prompt]
        text_emb = model.get_learned_conditioning(prompts)

        if args.mode == "base":
            cond = {"c_crossattn": [text_emb], "fps": fps}
        elif args.mode == "i2v":
            cond_images = load_image_batch(
                cond_inputs_rank[idx_s:idx_e], (args.height, args.width)
            )
            cond_images = cond_images.to(model.device)
            img_emb = model.get_image_embeds(cond_images)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            cond = {"c_crossattn": [imtext_cond], "fps": fps}
        else:
            raise NotImplementedError

        ## inference
        batch_samples = batch_ddim_sampling_freenoise(
            model,
            cond,
            noise_shape,
            args.n_samples,
            args.ddim_steps,
            args.ddim_eta,
            args.unconditional_guidance_scale,
            args=args,
            x_T_total=x_T_total,
        )

        out_path = "/tmp/output.mp4"
        vid_tensor = batch_samples[0]
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w

        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(args.n_samples))
            for framesheet in video
        ]  # [3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(
            out_path,
            grid,
            fps=args.savefps,
            video_codec="h264",
            options={"crf": "10"},
        )
        return Path(out_path)
