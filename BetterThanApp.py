import torch
import numpy as np
from torchvision import transforms

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class AppReplacement:

    def __init__(self, image, args, model, xcoords, ycoords, pos):

        # x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        # y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        self.pack(fill="both", expand=True)
        self.filename = ''
        self.filenames = []
        self.current_file_index = 0
        self.xcoords = xcoords
        self.ycoords = ycoords
        self.pos = pos
        self.image = image

        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = args.limit_longest_size
        self.add_click(self, xcoords, ycoords, pos)

    def add_click(self, xcoords, ycoords, pos):
        for z in range(len(pos)):
            tmp_bool = pos[z - 1] == 1
            click = clicker.Click(is_positive=tmp_bool,
                                  coords=(ycoords[z - 1], xcoords[z - 1]))  # I THINK THIS LINE IS VERY IMPORTANT
            self.clicker.add_click(click)
            print(click.coords)
            pred = self.predictor.get_prediction(self.clicker)
            torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

    def set_image(self, image):
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

        self.image = image
        self.image_nd = input_transform(image).to(self.device)
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.get_visualization(self, alpha_blend=0.5, click_radius=0.1)

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        if self.probs_history:
            results_mask_for_vis[self.current_object_prob > self.prob_thresh] = self.object_count + 1

        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis
