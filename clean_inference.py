import torch
from PIL import Image

from app.ClickSEG.isegm.inference import utils
from app.ClickSEG.interactive_demo.controller import InteractiveController

import io
from typing import Dict
from torch.serialization import _maybe_decode_ascii, _get_restore_location, StorageType
def my_load(zip_file, map_location, pickle_module, pickle_file='data.pkl', **pickle_load_args):
    restore_location = _get_restore_location(map_location)

    loaded_storages = {}

    def load_tensor(dtype, numel, key, location):
        name = f'data/{key}'

        storage = zip_file.get_storage_from_record(name, numel, torch.UntypedStorage).storage().untyped()
        # TODO: Once we decide to break serialization FC, we can
        # stop wrapping with TypedStorage
        loaded_storages[key] = torch.storage.TypedStorage(
            wrap_storage=restore_location(storage, location),
            dtype=dtype)

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert typename == 'storage', \
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, numel = data
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        if key not in loaded_storages:
            nbytes = numel * torch._utils._element_size(dtype)
            load_tensor(dtype, nbytes, key, _maybe_decode_ascii(location))

        return loaded_storages[key]

    load_module_mapping: Dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        'torch.tensor': 'torch._tensor'
    }

    # Need to subclass Unpickler instead of directly monkey-patching the find_class method
    # because it's marked readonly in pickle.
    # The type: ignore is because mypy can't statically determine the type of this class.
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # Lets us override the imports that pickle uses when unpickling an object.
        # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
        def find_class(self, mod_name, name):
            if type(name) is str and 'Storage' in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            mod_name = load_module_mapping.get(mod_name, mod_name)
            if mod_name.startswith('isegm'):
              mod_name = 'app.ClickSEG.' + mod_name
            return super().find_class(mod_name, name)

    # Load the data (which may in turn use `persistent_load` to load tensors)
    data_file = io.BytesIO(zip_file.get_record(pickle_file))

    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    torch._utils._validate_loaded_sparse_tensors()

    return result

torch.serialization._load = my_load

def load_controller(logger=None):
    if logger is not None:
        logger.info('Loading focalclick controller...')
    # checkpoint_path = utils.find_checkpoint('/code/app/weights/focalclick_models/', 'combined_segformerb3s2.pth')
    checkpoint_path = utils.find_checkpoint('app/weights/focalclick_models/', 'combined_segformerb3s2.pth')

    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = utils.load_is_model(checkpoint_path, device, cpu_dist_maps=True)
    controller = InteractiveController(model, device,
                                                predictor_params={'brs_mode': 'NoBRS', 'zoom_in_params': {'skip_clicks':-1, 'target_size': (448, 448)}},
                                                update_image_callback=None)
    def update_image_callback(reset_canvas=True):
      img = controller.get_visualization(0.5, 5)
    #   Image.fromarray(img).save('/vol/out.png')  # when in container
      Image.fromarray(img).save('out.png')

    controller.update_image_callback = update_image_callback
    return controller




