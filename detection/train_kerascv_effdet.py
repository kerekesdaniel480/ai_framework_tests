# train_kerascv_effdet.py
import os
import argparse
import tensorflow as tf
import keras_cv
from tf_coco_loader import load_coco_records, make_tf_dataset

def set_cpu_threads(n):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU
    tf.config.threading.set_intra_op_parallelism_threads(n)
    tf.config.threading.set_inter_op_parallelism_threads(n)

def build_model(img_size=512, num_classes=1):
    backbone = keras_cv.models.ResNet50Backbone(
        input_shape=(img_size, img_size, 3),
        include_top=False,
    )
    model = keras_cv.models.RetinaNet(
        num_classes=num_classes,
        backbone=backbone,
        bounding_box_format="xyxy",
        input_shape=(img_size, img_size, 3),
    )
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_json', required=True)
    p.add_argument('--train_images', required=True)
    p.add_argument('--val_json', required=False)
    p.add_argument('--val_images', required=False)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output_dir', default='outputs')
    args = p.parse_args()

    set_cpu_threads(args.threads)
    os.makedirs(args.output_dir, exist_ok=True)

    train_records = load_coco_records(args.train_json, args.train_images)
    train_ds = make_tf_dataset(train_records, img_size=args.img_size, batch_size=args.batch_size, shuffle=True)

    val_ds = None
    if args.val_json and args.val_images:
        val_records = load_coco_records(args.val_json, args.val_images)
        val_ds = make_tf_dataset(val_records, img_size=args.img_size, batch_size=1, shuffle=False)

    model = build_model(img_size=args.img_size, num_classes=1)

    # compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        box_loss=tf.keras.losses.Huber(),  # bbox regression loss
        classification_loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # classification loss
    )

    # Callbacks
    ckpt_path = os.path.join(args.output_dir, "ckpt_epoch_{epoch:02d}.weights.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_weights_only=True, save_freq='epoch'),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    steps_per_epoch = max(1, len(train_records) // args.batch_size)
    validation_steps = None
    if val_ds:
        validation_steps = max(1, len(val_records) // 1)

    model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

if __name__ == '__main__':
    main()