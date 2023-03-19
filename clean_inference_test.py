import numpy as np
from PIL import Image

from app.ClickSEG.clean_inference import load_controller

def clean_inference_test():
  img = Image.open('../test_imgs/GrabCut_153093.jpg')
  img = img.convert("RGB")
  img.save('../results/img.png')
  clicks = [(218, 148, True), (299, 171, False)]

  modelname = 'focalclick'
  controller = load_controller()
  assert controller is not None

  # inference
  controller.set_image(np.array(img))

  for i, click in enumerate(clicks):
    controller.add_click(click[0], click[1], click[2])
    result = controller.result_mask
    Image.fromarray(np.array(0 < result)).save(f'../results/result_{modelname}_{i}.png')


if __name__ == '__main__':
    clean_inference_test()
