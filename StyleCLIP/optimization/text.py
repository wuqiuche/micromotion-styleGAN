import argparse
import math
import os
import numpy as np
import random
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch
import torchvision
from torch import optim
from tqdm import tqdm
import clip

from StyleCLIP.criteria.clip_loss import CLIPLoss
from StyleCLIP.criteria.id_loss import IDLoss
from StyleCLIP.models.stylegan2.model import Generator
from StyleCLIP.models.stylegan2.model import Generator


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def get_text_latent(args,
                    text,
                    degrees=[
                        "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%",
                        "90%", "100%"
                    ],
                    seed=0,
                    exp_name=''):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    latents = []
    temp_dir = 'text/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    with torch.no_grad():
        g_ema = Generator(1024, 512, 8)
        g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
        g_ema.eval()
        g_ema = g_ema.cuda()
        model, _ = clip.load("ViT-B/32", device="cuda")
        mean_latent = g_ema.mean_latent(4096)
        upsample_func = torch.nn.Upsample(1024)

        # set image from a random latent
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        _, latent_code_init, _ = g_ema([latent_code_init_not_trunc],
                                       return_latents=True,
                                       truncation=args.truncation,
                                       truncation_latent=mean_latent)

        latents.append(latent_code_init.detach().clone().squeeze(0))

        # start editing
        img_orig, _ = g_ema([latent_code_init],
                            input_is_latent=True,
                            randomize_noise=False)
        img_orig = upsample_func(img_orig)

    clip_loss = CLIPLoss(args).cuda()
    id_loss = IDLoss(args).cuda()
    torchvision.utils.save_image(img_orig,
                                 f"{temp_dir}/{exp_name}_org.png",
                                 normalize=True,
                                 range=(-1, 1))

    for degree in tqdm(degrees):
        full_text = eval("f'{}'".format(text))
        text_inputs = torch.cat([clip.tokenize(full_text)]).cuda()
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

        optimizer = optim.Adam([latent], lr=args.lr)
        # pbar = tqdm(range(args.step))

        for i in range(args.step):
            random.seed(seed+i)
            torch.manual_seed(seed+i)
            torch.cuda.manual_seed(seed+i)
            torch.cuda.manual_seed_all(seed+i)
            np.random.seed(seed+i)
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr

            img_gen, _ = g_ema([latent],
                               input_is_latent=True,
                               randomize_noise=False,
                               input_is_stylespace=args.work_in_stylespace)

            c_loss = clip_loss(img_gen, text_inputs)

            if args.id_lambda > 0:
                i_loss = id_loss(img_gen, img_orig)[0]
            else:
                i_loss = 0

            if args.mode == "edit":
                if args.work_in_stylespace:
                    l2_loss = sum([
                        ((latent_code_init[c] - latent[c])**2).sum()
                        for c in range(len(latent_code_init))
                    ])
                else:
                    l2_loss = ((latent_code_init - latent)**2).sum()
                loss = c_loss + args.l2_lambda * l2_loss + args.id_lambda * i_loss
            else:
                loss = c_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            img_gen, _ = g_ema([latent.detach()],
                               input_is_latent=True,
                               randomize_noise=False,
                               input_is_stylespace=args.work_in_stylespace)
            torchvision.utils.save_image(img_gen,
                                         f"{temp_dir}/{exp_name}_{degree}.png",
                                         normalize=True,
                                         range=(-1, 1))
        latents.append(latent.detach().clone().squeeze(0))
        torch.cuda.empty_cache()

    return latents
