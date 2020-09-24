import argparse
import BetterThanApp
import torch
import cv2
import sys

from isegm.utils import exp
from isegm.inference import utils


def main():
    args, cfg = parse_args()

    torch.backends.cudnn.deterministic = True
    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True, norm_radius=args.norm_radius)

    x_coords = (300, 300, 150)
    y_coords = (300, 400, 300)
    is_pos = (1, 1, 0)
    img_pth = '/Users/jason/Desktop/img_1205.jpg'
    image = cv2.cvtColor(cv2.imread(img_pth), cv2.COLOR_BGR2RGB)
    print(type(image))
    print(sys.getsizeof(image))

    var1 = BetterThanApp.AppReplacement(image, args, model, x_coords, y_coords, is_pos)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Use only CPU for inference.')

    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')

    parser.add_argument('--norm-radius', type=int, default=260)

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg


if __name__ == '__main__':
    main()
