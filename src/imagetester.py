import os
import math

import utility
from data import common

import torch
import cv2

from tqdm import tqdm

class ImageTester():
    def __init__(self, args, my_model, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.model = my_model

        self.filename, _ = os.path.splitext(os.path.basename(args.dir_demo))

    def test(self):
        torch.set_grad_enabled(False)

        self.ckp.write_log('\nEvaluation on single image:')
        self.model.eval()

        timer_test = utility.timer()
        for idx_scale, scale in enumerate(self.scale):
            img = cv2.imread(self.args.dir_demo)
            lr, = common.set_channel(img, n_channels=self.args.n_colors)
            lr, = common.np2Tensor(lr, rgb_range=self.args.rgb_range)
            lr, = self.prepare(lr.unsqueeze(0))
            sr = self.model(lr, idx_scale)
            sr = utility.quantize(sr, self.args.rgb_range).squeeze(0)
            normalized = sr * 255 / self.args.rgb_range
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            cv2.imwrite(args.dest_file,ndarr)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]