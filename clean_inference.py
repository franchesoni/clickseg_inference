import torch
from PIL import Image

from isegm.inference import utils
from interactive_demo.controller import InteractiveController

def load_controller():
    checkpoint_path = utils.find_checkpoint('../weights/focalclick_models/', 'combined_segformerb3s2.pth')

    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = utils.load_is_model(checkpoint_path, device, cpu_dist_maps=True)
    controller = InteractiveController(model, device,
                                                predictor_params={'brs_mode': 'NoBRS', 'zoom_in_params': {'skip_clicks':-1, 'target_size': (448, 448)}},
                                                update_image_callback=None)
    def update_image_callback(reset_canvas=True):
      img = controller.get_visualization(0.5, 5)
      Image.fromarray(img).save('../results/out.png')

    controller.update_image_callback = update_image_callback
    return controller


