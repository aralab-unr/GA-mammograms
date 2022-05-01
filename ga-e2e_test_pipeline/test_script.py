import image_clf_train

TRAIN_DIR = "/content/drive/MyDrive/GA-mammograms/test_run_images/train"
VAL_DIR = "/content/drive/MyDrive/GA-mammograms/test_run_images/val"
TEST_DIR = "/content/drive/MyDrive/GA-mammograms/test_run_images/test"
BEST_MODEL = "/content/drive/MyDrive/GA-mammograms/ddsm_train/ddsm_vgg16_s10_[512-512-1024]x2_hybrid.h5"

# original values run
image_clf_train.run(
            train_dir=TRAIN_DIR,
            val_dir=VAL_DIR,
            test_dir=TEST_DIR,
            resume_from=BEST_MODEL,
            img_size=[1152, 896],
            rescale_factor=0.003891,
            featurewise_mean=44.33,
            patch_net='resnet50',
            block_type='resnet',
            batch_size=2, #tweak this parameter for better performance
            all_layer_epochs=100, #tweak this parameter for better performance
            load_val_ram=True,
            load_train_ram=True,
            weight_decay = 0.0001,
            weight_decay2 = 0.0001,
            init_lr = 0.01,
            all_layer_multiplier = 0.1,
            pos_cls_weight = 1.0,
            neg_cls_weight = 1.0,
            lr_patience=2,
            es_patience=10,
            augmentation=False,
            nb_epoch = 0,
            best_model = 'best_model.h5'
)

# GA values run
# image_clf_train.run(
#             train_dir=TRAIN_DIR,
#             val_dir=VAL_DIR,
#             test_dir=TEST_DIR,
#             resume_from=BEST_MODEL,
#             img_size=[1152, 896],
#             rescale_factor=0.003891,
#             featurewise_mean=44.33,
#             patch_net='resnet50',
#             block_type='resnet',
#             batch_size=2, #tweak this parameter for better performance
#             all_layer_epochs=100, #tweak this parameter for better performance
#             load_val_ram=True,
#             load_train_ram=True,
#             weight_decay = 0.302,
#             weight_decay2 = 0.693,
#             init_lr = 0.043,
#             all_layer_multiplier = 0.114,
#             pos_cls_weight = 0.227,
#             neg_cls_weight = 0.708,
#             lr_patience=2,
#             es_patience=10,
#             augmentation=False,
#             nb_epoch = 0,
#             best_model = 'best_model.h5'
# )