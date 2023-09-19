import torch
import torchvision


AUGMENT_OPS = {
    "random_resized_crop": torchvision.transforms.RandomResizedCrop,
    "color_jitter": torchvision.transforms.ColorJitter,
}


def get_augment_transform(augment_kwargs):
    transforms = []
    for op in augment_kwargs["augment_order"]:
        transform = AUGMENT_OPS[op](**augment_kwargs[op])
        transforms.append(transform)

    transforms = torchvision.transforms.Compose(transforms)
    # transforms = torch.nn.Sequential(*transforms)
    # transforms = torch.jit.script(transforms)

    return transforms
