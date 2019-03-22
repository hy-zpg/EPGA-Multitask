main model: models including light network and popular network
            1. ligth: mini_xception
            2. popular: vgg, resnet and senet pretrained on VGGFace

parameters:
model_name: light or popular
input_type: (64,64,1) (224,224,3)
task_type:1-12, different tasks
## classes
emotion_classes
pose_classes
age_classes
gender_classes
## network method to alleviate overfit
is_droput
is_bn
weights_decay
