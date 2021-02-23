import tensorflow as tf
import numpy as np
import argparse

from deep_models import UNet
from data import brain_dataset
from metrics import dice_metrics
from params import train_params


def main(args):
    params = train_params()
    val_ds, _ = brain_dataset(f'{args.dataset_val_path}/{args.axis}', image_size=params[args.axis])
    val_ds = val_ds.batch(args.batch_size)

    model_base_path = args.model_path
    model = UNet()
    model.load_weights(model_base_path)

    mean_dice = []

    for i, (image, label) in enumerate(val_ds):
        output = model(image, training=False)
        dice = dice_metrics(label, output)
        print(f'Dice score: {dice}')
        mean_dice.append(dice)

    print(np.mean(mean_dice))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-val-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--axis', type=str, default='x')
    parser.add_argument('--batch-size', type=int, default=16)

    args, _ = parser.parse_known_args()
    main(args)
