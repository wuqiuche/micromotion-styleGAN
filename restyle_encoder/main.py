from argparse import Namespace
import os
import sys
import torch
import torchvision.transforms as transforms
import dlib

from models.psp import pSp
from models.psp import pSp
from scriptsLocal.align_faces_parallel import align_face
from utils.inference_utils import run_batch_latent

sys.path.append("./restyle_encoder")


def run_alignment(image_path):
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system(
            'wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        )
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def get_avg_image(net):
    avg_image = net(net.latent_avg.unsqueeze(0),
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach()
    return avg_image


def get_org_latent(image_path):
    model_path = "restyle_encoder/pretrained_models/restyle_psp_ffhq_encode.pt"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    with torch.no_grad():
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        net = pSp(opts).cuda().eval()
        input_image = run_alignment(image_path).convert('RGB')
        input_image.resize((256, 256))
        transformed_image = transform(input_image)
        opts.n_iters_per_batch = 5
        opts.resize_outputs = False
        avg_image = get_avg_image(net)
        latents = run_batch_latent(
            transformed_image.unsqueeze(0).cuda(), net, opts, avg_image)
    return latents[0][4]  # the inverted results from the last iteration
