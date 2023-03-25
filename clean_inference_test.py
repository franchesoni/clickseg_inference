import cv2
import numpy as np
from PIL import Image

from app.ClickSEG.clean_inference import load_controller
from app.ClickSEG.isegm.inference.clicker import Clicker

def clean_inference_test():
  img = Image.open('img.png')
  img = img.convert("RGB")
  img.save('img.png')
  clicks = [(218, 148, True), (299, 171, False)]

  modelname = 'focalclick'
  controller = load_controller()
  assert controller is not None

  # inference
  controller.set_image(np.array(img))

  for i, click in enumerate(clicks):
    controller.add_click(click[0], click[1], click[2])
    result = controller.result_mask
    Image.fromarray(np.array(0 < result)).save(f'result_{modelname}_{i}.png')



def inference_with_mask():
  img = Image.open('img_2273.png')
  img = img.convert("RGB")
  img.save('img.png')
  target = Image.open('target.png')

  external_clicker = Clicker(gt_mask=np.array(target))

  modelname = 'focalclick'
  controller = load_controller()
  assert controller is not None

  # inference
  controller.set_image(np.array(img))

  pred_mask = np.zeros_like(np.array(target))
  for i, click in enumerate(range(22)):
    external_clicker.make_next_click(pred_mask)
    clicks_so_far = external_clicker.get_clicks()
    last_click = clicks_so_far[-1]
    (y, x) = last_click.coords
    is_positive = last_click.is_positive
    controller.add_click(x, y, is_positive)
    pred_mask = controller.result_mask
    Image.fromarray(np.array(0 < pred_mask)).save(f'result_{modelname}_{i}.png')
    breakpoint()

if __name__ == '__main__':
    # clean_inference_test()
    inference_with_mask()
