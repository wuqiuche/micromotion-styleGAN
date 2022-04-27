import os
import warnings
from argparse import Namespace
import argparse
import torch
import torchvision.transforms as transforms
import dlib
import imageio
from skimage import img_as_ubyte
from tqdm import tqdm

from restyle_encoder.models.psp import pSp
from restyle_encoder.scriptsLocal.align_faces_parallel import align_face
from restyle_encoder.utils.inference_utils import run_batch_latent
from StyleCLIP.optimization.text import get_text_latent
from StyleCLIP.optimization.pca import get_pca_latent
from StyleCLIP.models.stylegan2.model import Generator
from StyleCLIP.optimization.mixin import mixin

warnings.filterwarnings("ignore")


def run_alignment(image_path, img_size=512):
    # Given an image with human face, download a model to align the human face to the center of the image.
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system(
            'wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        )
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    try:
        aligned_image = align_face(filepath=image_path,
                                   predictor=predictor,
                                   output_size=img_size,
                                   transform_size=img_size)
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image
    except:
        return None


def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def get_img_latent(net, opts, transform, image_path):
    # compute the latent given an image (i.e. the inverter)
    with torch.no_grad():
        img = run_alignment(image_path, img_size=256)
        if img is None:
            return None
        input_image = img.convert('RGB')
        input_image.resize((256, 256))
        transformed_image = transform(input_image)
        avg_image = get_avg_image(net)
        latent = run_batch_latent(
            transformed_image.unsqueeze(0).cuda(), net, opts, avg_image)
    latent = latent[0][4].detach().clone()
    return latent


def main(args):
    input_img = args.input
    scale = args.scale
    cate = args.category

    if cate in ["smile", "angry", "aging", "eyesClose", "headsTurn"]:
        # Using pre-computed latents
        inverter_ckpt = args.inverter or "restyle_encoder/pretrained_models/restyle_psp_ffhq_encode.pt"
        generator_ckpt = args.generator or "StyleCLIP/stylegan2-ffhq-config-f.pt"

        inverter = torch.load(inverter_ckpt, map_location='cuda')
        opts = inverter['opts']
        opts['checkpoint_path'] = inverter_ckpt
        opts = Namespace(**opts)
        net = pSp(opts).cuda().eval()
        opts.n_iters_per_batch = 5
        opts.resize_outputs = False

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # inverting
        print(f'computing latent ...')
        latents = get_img_latent(net, opts, transform, input_img)
        if latents is None:
            print("Cannot identify person face in this image!")
            return None

        # computation
        latents_begin = latents.unsqueeze(0)
        d_latent = torch.load("pregen_latents/%s/d_latent" % (cate)) * scale
        latents = [(latents_begin + i / 100 * d_latent) for i in range(100)]

        # generating
        print(f'generating frames ...')
        with torch.no_grad():
            images = []
            g_ema = Generator(1024, 512, 8)
            g_ema.load_state_dict(torch.load(generator_ckpt)["g_ema"],
                                  strict=False)
            g_ema.cuda().eval()
            for frame in range(len(latents)):
                img_gen_full, _ = g_ema([latents[frame]],
                                        input_is_latent=True,
                                        randomize_noise=False,
                                        input_is_stylespace=False)
                img_gen_full = (img_gen_full - torch.min(img_gen_full)) / (
                    torch.max(img_gen_full) - torch.min(img_gen_full))
                img_gen_full = img_as_ubyte(
                    img_gen_full.squeeze(0).detach().cpu())
                images.append(img_gen_full.swapaxes(0, 2).swapaxes(0, 1))
            imageio.mimsave(os.path.join("save",
                                         f"{input_img[:-4]}_{cate}_mix.mp4"),
                            images,
                            fps=30)
        return None
    if cate == "custom":
        # Using newly computed latents from scratch (template).
        image_path = args.input
        if args.template == '':
            print("Please specify text template in the custom mode.")
            return None

        text = args.template
        degrees = [
            "no", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%",
            "90%", "100%"
        ]

        num_latent = args.num_latent
        seed_list = [args.seed + basis for basis in range(num_latent)]
        scale_list = [8]
        pca_type = args.pca_type
        step = 40

        for seed in tqdm(seed_list):
            num_frames = 90
            fps = 60
            create_video = True
            args = {
                "ckpt": "StyleCLIP/stylegan2-ffhq-config-f.pt",
                "stylegan_size": 1024,
                "lr_rampup": 0.05,
                "pca_type": pca_type,
                "lr": 0.1,
                "step": step,
                "mode": 'edit',
                "l2_lambda": 0.008,
                "id_lambda": 0.005,
                'work_in_stylespace': False,
                "latent_path": None,
                "truncation": 0.7,
                "save_intermediate_image_every": 1 if create_video else 20,
                "results_dir": "results",
                "ir_se50_weights": "StyleCLIP/model_ir_se50.pth"
            }  # we follow the parameters in StyleCLIP
            latents_dir = 'latent/'
            latents_pca_dir = 'latent_pca/'
            args = Namespace(**args)
            text_save = text.replace(" ", "_")
            exp_name = f'{text_save}_s{seed}_pre_pca'
            exp_name_pca = f'{text_save}_s{seed}_pca'

            if not os.path.exists(latents_dir):
                os.makedirs(latents_dir, exist_ok=True)
            if not os.path.exists(latents_pca_dir):
                os.makedirs(latents_pca_dir, exist_ok=True)

            if os.path.isfile(os.path.join(latents_dir, exp_name)):
                print(f'loading {os.path.join(latents_dir, exp_name)}')
                latents_text = torch.load(
                    os.path.join(latents_dir, exp_name))
            else:
                print(
                    f'computing latent {os.path.join(latents_dir, exp_name)}'
                )
                latents_text = get_text_latent(args=args,
                                               text=text,
                                               degrees=degrees,
                                               seed=seed,
                                               exp_name=exp_name)
                torch.save(latents_text,
                           os.path.join(latents_dir, exp_name))
            if os.path.isfile(os.path.join(latents_pca_dir, exp_name_pca)):
                print(
                    f'loading {os.path.join(latents_pca_dir, exp_name_pca)}'
                )
                latents_text_pca = torch.load(
                    os.path.join(latents_pca_dir, exp_name_pca))
            else:
                print(
                    f'computing pca latent {os.path.join(latents_pca_dir, exp_name_pca)}'
                )
                latents_text_pca = get_pca_latent(args=args,
                                                  latents=latents_text,
                                                  text=text,
                                                  degrees=degrees,
                                                  exp_name=exp_name)
                torch.save(latents_text_pca,
                           os.path.join(latents_pca_dir, exp_name_pca))

            g_ema = Generator(args.stylegan_size, 512, 8)
            g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"],
                                  strict=False)
            g_ema.cuda().eval()

            model_path = "restyle_encoder/pretrained_models/restyle_psp_ffhq_encode.pt"
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            ckpt = torch.load(model_path, map_location='cpu')
            opts = ckpt['opts']
            opts['checkpoint_path'] = model_path
            opts = Namespace(**opts)
            net = pSp(opts).cuda().eval()
            opts.n_iters_per_batch = 5
            opts.resize_outputs = False

            imgname = os.path.splitext(os.path.basename(image_path))[0]
            print(f"image: {image_path}")
            latent_img = get_img_latent(net, opts, transform,
                                        image_path)
            if latent_img is None:
                continue
            for scale in scale_list:
                exp_name_img = f'{imgname}_{text_save}'
                mixin(args=args,
                      latents_text_pca=latents_text_pca,
                      latent_img=latent_img,
                      scale=scale,
                      num_frames=num_frames,
                      fps=fps,
                      exp_name=exp_name_img)
        return None

    # Otherwise, category is incorrectly specified.
    print("Category undefined.")
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category',
                        type=str,
                        default='smile',
                        choices=[
                            'smile', 'angry', 'aging', 'eyesClose',
                            'headsTurn', 'custom'
                        ])
    parser.add_argument('--scale', type=int, default=5)
    parser.add_argument('--inverter', type=str)
    parser.add_argument('--generator', type=str)
    parser.add_argument('--template', type=str, default='')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pca_type', type=str, default='rpca')
    parser.add_argument('--num_latent', type=int, default=1)
    parser.add_argument('--input',
                        type=str,
                        default='examples/01.jpg')
    args = parser.parse_args()
    main(args)
