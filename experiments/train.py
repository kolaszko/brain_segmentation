import tensorflow as tf
from datetime import datetime
import argparse

from data import brain_dataset
from deep_models import UNet
from metrics import dice_metrics, dice_loss
from params import train_params


def main(args):
    model = UNet()

    params = train_params()
    train_ds, _ = brain_dataset(f'{args.dataset_train_path}/{args.axis}', image_size=params[args.axis])
    val_ds, _X = brain_dataset(f'{args.dataset_train_path}/{args.axis}', image_size=params[args.axis])  #

    train_ds = train_ds.batch(args.batch_size)
    val_ds = val_ds.batch(args.batch_size)

    logdir = f'logs/scalars/{args.axis}/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    file_writer = tf.summary.create_file_writer(logdir + '/metrics')
    file_writer.set_as_default()

    lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=params['lr'],
                                                                  decay_steps=params['decay_steps'], decay_rate=params['decay_rate'], staircase=True)
    optimizer = tf.keras.optimizers.Adam(lr_scheduler)
    metrics = tf.keras.metrics.BinaryCrossentropy()

    @tf.function
    def train(images, labels):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = dice_loss(labels, output)
            metric = metrics(labels, output)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss, metric

    @tf.function
    def validate(images, labels):
        output = model(images, training=False)
        loss = dice_loss(labels, output)
        metric = metrics(labels, output)

        return loss, metric

    step_train_offset = 0
    step_val_offset = 0

    for e in range(args.epochs):
        loss_mean = []
        last_loss = 10e6

        step_counter = 0

        for step, (image, labels) in enumerate(train_ds):
            loss, metric = train(image, labels)

            if step % 100 == 0:
                print(f'Train step: {step}')
                tf.summary.scalar('train/crossentropy', metric, step=step + step_train_offset)
                tf.summary.scalar('train/dice', loss, step=step + step_train_offset)
                tf.summary.scalar('learning_rate', optimizer._decayed_lr(tf.float32), step=step + step_train_offset)
                step_counter += step

            print(f"Dice train: {loss}")

        step_train_offset += step_counter

        for step, (image, labels) in enumerate(val_ds):
            loss, metric = validate(image, labels)

            if step % 100 == 0:
                print(f'Val step: {step}')
                tf.summary.scalar('val/crossentropy', metric, step=step + step_val_offset)
                tf.summary.scalar('val/dice', loss, step=step + step_val_offset)
                step_val_offset += step

            loss_mean.append(loss)
            print(f"Dice validation: {loss}")

        if last_loss > (mean_loss := tf.reduce_mean(loss_mean)):
            last_loss = mean_loss
            model.save_weights(f'{logdir}/model_{e}.save')
            print("Saved model")

        print("===============")
        print(f"Mean Loss: {tf.reduce_mean(loss_mean)}")
        print(f"Epoch: {e}")
        print("===============")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-train-path', type=str, required=True)
    parser.add_argument('--dataset-val-path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--axis', type=str, default='x')

    args, _ = parser.parse_known_args()
    main(args)
