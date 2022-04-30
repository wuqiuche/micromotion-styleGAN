import os
import tempfile
from argparse import Namespace
from skimage import img_as_ubyte
from base64 import b64encode
import imageio
import torch
import torchvision.transforms as transforms

from cog import BasePredictor, Path, Input
from StyleCLIP.models.stylegan2.model import Generator
from restyle_encoder.models.psp import pSp
from main import get_img_latent


class Predictor(BasePredictor):
    def setup(self):
        inverter_ckpt = "restyle_encoder/pretrained_models/restyle_psp_ffhq_encode.pt"
        self.generator_ckpt = "StyleCLIP/stylegan2-ffhq-config-f.pt"

        inverter = torch.load(inverter_ckpt, map_location="cuda")
        self.opts = inverter["opts"]
        self.opts["checkpoint_path"] = inverter_ckpt
        self.opts = Namespace(**self.opts)
        self.net = pSp(self.opts).cuda().eval()
        self.opts.n_iters_per_batch = 5
        self.opts.resize_outputs = False

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def predict(
        self,
        image: Path = Input(
            description="Input image. Image will be aligned and resized to 256x256.",
        ),
        micromotion: str = Input(
            choices=["smile", "angry", "aging", "eyesClose", "headsTurn"],
            default="eyesClose",
            description="Choose a micromotion.",
        ),
        scale: int = Input(
            default=5,
            ge=1,
            le=10,
        ),
    ) -> Path:
        print(type(micromotion))

        print(f"computing latent ...")
        latents = get_img_latent(self.net, self.opts, self.transform, str(image))
        if latents is None:
            print("Cannot identify person face in this image!")
            return None

        # computation
        latents_begin = latents.unsqueeze(0)
        d_latent = torch.load("pregen_latents/%s/d_latent" % (str(micromotion).split('.')[-1])) * scale
        latents = [(latents_begin + i / 100 * d_latent) for i in range(100)]

        # generating
        print(f"generating frames ...")
        with torch.no_grad():
            images = []
            g_ema = Generator(1024, 512, 8)
            g_ema.load_state_dict(
                torch.load(self.generator_ckpt)["g_ema"], strict=False
            )
            g_ema.cuda().eval()
            for frame in range(len(latents)):
                img_gen_full, _ = g_ema(
                    [latents[frame]],
                    input_is_latent=True,
                    randomize_noise=False,
                    input_is_stylespace=False,
                )
                img_gen_full = (img_gen_full - torch.min(img_gen_full)) / (
                    torch.max(img_gen_full) - torch.min(img_gen_full)
                )
                img_gen_full = img_as_ubyte(img_gen_full.squeeze(0).detach().cpu())
                images.append(img_gen_full.swapaxes(0, 2).swapaxes(0, 1))

        out_path = Path(tempfile.mkdtemp()) / "output.mp4"
        imageio.mimsave(str(out_path), images, fps=30)

        return out_path
