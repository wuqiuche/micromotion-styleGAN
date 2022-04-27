import os
import torch
import torchvision
import numpy as np
import torch.nn.functional as F

from StyleCLIP.models.stylegan2.model import Generator


class RPCA_gpu:
    """ low-rank and sparse matrix decomposition via RPCA [1] with CUDA capabilities """
    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = torch.zeros_like(self.D)
        self.Y = torch.zeros_like(self.D)
        self.mu = mu or (np.prod(self.D.shape) /
                         (4 * self.norm_p(self.D, 2))).item()
        self.mu_inv = 1 / self.mu
        self.lmbda = lmbda or 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return torch.sum(torch.pow(M, p))

    @staticmethod
    def shrink(M, tau):
        return torch.sign(M) * F.relu(
            torch.abs(M) - tau)  # hack to save memory

    def svd_threshold(self, M, tau):
        U, s, V = torch.svd(M, some=True)
        return torch.mm(U, torch.mm(torch.diag(self.shrink(s, tau)), V.t()))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        i, err = 0, np.inf
        Sk, Yk, Lk = self.S, self.Y, torch.zeros_like(self.D)
        _tol = tol or 1e-7 * self.norm_p(torch.abs(self.D), 2)
        while err > _tol and i < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk,
                                    self.mu_inv)
            Sk = self.shrink(self.D - Lk + (self.mu_inv * Yk),
                             self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(torch.abs(self.D - Lk - Sk), 2) / self.norm_p(
                self.D, 2)
            i += 1
            # if (i % iter_print) == 0 or i == 1 or i > max_iter or err <= _tol:
            #     print(f'Iteration: {i}; Error: {err:0.4e}')
        self.L, self.S = Lk, Sk
        return Lk, Sk


def get_pca_latent(args, latents, text, degrees, exp_name):
    save_dir = 'text_pca/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    text_latents = []
    new_latents = [torch.zeros_like(l) for l in latents]
    for i in range(latents[0].shape[0]):
        new_tensor = torch.zeros(0, 512).to("cuda")
        for j in range(len(new_latents)):
            new_tensor = torch.cat((new_tensor, latents[j][i].reshape(1, -1)),
                                   dim=0)
        # results = torch.pca_lowrank(new_tensor, q=4, center=False)
        solver = RPCA_gpu(new_tensor)
        new_tensor_lowrank, _ = solver.fit()
        results = torch.pca_lowrank(new_tensor_lowrank, q=4, center=False)

        tmp = torch.matmul(results[0], torch.diag(results[1]))
        tmp = torch.matmul(tmp, torch.transpose(results[2], 0, 1))
        for j in range(len(new_latents)):
            new_latents[j][i] = tmp[j]

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    for i, degree in enumerate(degrees):
        text_latents.append(torch.unsqueeze(new_latents[i], 0))
        img_gen, _ = g_ema([torch.unsqueeze(new_latents[i], 0)],
                           input_is_latent=True,
                           randomize_noise=False,
                           input_is_stylespace=args.work_in_stylespace)
        torchvision.utils.save_image(img_gen,
                                     f"{save_dir}/{exp_name}_{degree}.png",
                                     normalize=True,
                                     range=(-1, 1))
    return text_latents
