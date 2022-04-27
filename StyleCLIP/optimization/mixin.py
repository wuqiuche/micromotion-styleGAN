import os
import torch
import torchvision
from skimage import img_as_ubyte

from StyleCLIP.models.stylegan2.model import Generator
import imageio


def mixin(args,
          latents_text_pca,
          latent_img,
          scale,
          num_frames=100,
          fps=60,
          exp_name=''):
    video_dir = 'results/'
    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)

    with torch.no_grad():
        latent_none = latents_text_pca[0]
        latent_full = latents_text_pca[-1]
        latent_zero = latent_img.unsqueeze(0)
        diff = latent_full - latent_none
        latents = [
            latent_zero + diff * (i / 100) * scale for i in range(0, 100)
        ]
        g_ema = Generator(args.stylegan_size, 512, 8)
        g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
        g_ema.eval()
        g_ema = g_ema.cuda()

        images = []
        for frame in range(num_frames):
            img_gen_full, _ = g_ema(
                [latents[frame]],
                input_is_latent=True,
                randomize_noise=False,
                input_is_stylespace=args.work_in_stylespace)
            img_gen_full = (img_gen_full - torch.min(img_gen_full)) / (
                torch.max(img_gen_full) - torch.min(img_gen_full))
            img_gen_full = img_as_ubyte(img_gen_full.squeeze(0).detach().cpu())
            images.append(img_gen_full.swapaxes(0, 2).swapaxes(0, 1))
        imageio.mimsave(os.path.join(video_dir, f"{exp_name}.gif"),
                        images,
                        fps=fps)
