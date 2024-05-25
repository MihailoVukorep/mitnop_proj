#!/usr/bin/env python3

# %% libs
from utils import *

# %% load set
print("loading datasets...")
images, labels, mapping = dataset_load_all()
mapping_n = len(mapping)

# %% dataset info
print("using datasets:")
print(f'images..: {images.shape}')
print(f'labels..: {labels.shape}')
print(f'mapping.: {mapping_n}')
print(f'bytes...: {bytes_human_readable(images.nbytes)}')
print()

# %% preprocess set for tf
print("preparing images for tensorflow...")
images_tf, labels_tf = prepdata(images, labels)
print("images prepared.")


# %% train val split

train_input, test_input, train_target, test_target = train_test_split(images_tf, labels_tf, test_size=0.2, random_state=42)

# %% create model
print("creating model:")
model = create_model()

# %% training basic model

num_epochs = 5
batch_size = 64

print("training model...")
history = model.fit(
    train_input,
    train_target,
    batch_size=batch_size,
    validation_data=(test_input, test_target),
    epochs=num_epochs,
    verbose=2
)
print("model trained.")

# %% save model
print("saving model...")
model.save(d_models('model_all.keras'))
print("model saved.")




# # %% training a super model
# num_epochs = 50
# batch_size = 64

# datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)

# def lr_scheduler(epoch, lr):
#     if epoch % 10 == 0 and epoch != 0:
#         lr = lr * 0.9
#     return lr

# lr_callback = LearningRateScheduler(lr_scheduler)
# print("training model...")
# history = model.fit(
#     datagen.flow(train_input, train_target, batch_size=batch_size),
#     steps_per_epoch=len(train_input) // batch_size,
#     validation_data=(test_input, test_target),
#     epochs=num_epochs,
#     callbacks=[lr_callback],
#     verbose=2
# )
# print("model trained.")

# # %% save model
# print("saving model...")
# model.save(d_models('model_all.keras'))
# print("model saved.")

# %%
