
## Preparation
### face-centric application
**Update from old [repo](https://github.com/dandynaufaldi/Agendernet)**

### Environment Setup
Using python 3.6, core libraries are :
- [dlib](https://github.com/davisking/dlib)
- tensorflow
- keras
- opencv

### Datasets
Datasets are saved in data/ directory
- [IMDB-Wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
- [Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
- [UTKFace](https://susanqq.github.io/UTKFace/)
- [FGNET](http://yanweifu.github.io/FG_NET_data/index.html)
- [FERPLUS](https://github.com/Microsoft/FERPlus)
- [SFEW](https://computervisiononline.com/dataset/1105138659)
- [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)
- [EXPW](http://home.ie.cuhk.edu.hk/~ccloy/download.html)


### Model
Here we use 2 model
#### MINI_XCEPTION 
We use InceptionV3 and MobileNetV2 from keras-application without it's classifier output and modify it to have 2 output layer. 
One for gender prediction with 2 nodes and another for age prediction with 101 nodes (represent age 0-100).
We treat age prediction as multiclass classification problem with it's output calculated from softmax regression as in [1] reference
#### VGGFACE
We follow default SSR-Net architecture with some modification. At the top we have 1 classifier block which then feed into 2 classifier block. In SSR-Net, we treat prediction as regression problem for both age and gender 

### Train
For InceptionV3 and MobileNetV2 we used pretrained weight from keras-application which was trained on ImageNet dataset. So, we do transfer learning on InceptionV3 and MobileNetV2. And, training from scratch on SSR-Net
Training was done in AWS P3 instance with Nvidia Tesla V100 with 10 fold cross-validation and 50 epoch each to check for model consistency and get weight file with best metrics
For each model, following input size is used:
- MINI_XCEPTION : 64 x 64 px
- VGGFACE: 224 x 224  x 3 px


## Running code
### data:
   * storing datasets
   * prepare images-labels pairs
   * generating csv file

### utils:
   stl: prepare data generator for single-task learning
   confusion: prepare data generator for multi-task learning with joint training method
   alternative: prepare data generator for single-task learning with alternative triaing method

### training scripts for different methods
#### alternative-mtl:
   * EAGP: emotion, age, gener, pose
   * runing script:
python ALTERNATIVE_EP.py {other similar function, ALTERNATIVE_EA.py, ALTERNATIVE_EGA.py}
$ dataset 
--dataset_emotion=expw 
--dataset_pose=aflw
$ model 
--epoch=64 
--model=vggFace  
--batch_size=32 
$ traiing tricks
--is_augmentation=False 
--is_dropout=False 
--is_bn=False 
--weights_decay=0 
--is_freezing=False 
--no_freezing_epoch=0 
$ multi-task weights
--P_loss_weights=1 
--E_loss_weights=1
$ preserve information method(distilling knowledge) && augment information method(pseudo labels)
--is_naive=False 
--is_distilled=False 
--is_pesudo=True 
--is_interpolation=False 
--interpolation_weights=0 
--selection_threshold=0.8
$ selection method 
--is_pesudo_confidence=True 
--is_pesudo_density=True 
--is_pesudo_distribution=True 
--cluster_k=3 

#### confusion-mtl
   * EAGP: emotion, age, gener, pose
   * runing script:
python CONFUSION_EPGA_multitask.py {other similar function,CONFUSION_EA_multitask.py}
$ dataset 
--dataset_emotion=expw 
--dataset_pose=aflw
$ model 
--epoch=64 
--model=vggFace 
--batch_size=32 
$ traiing tricks
--is_augmentation=False 
--is_dropout=False 
--is_bn=False 
--weights_decay=0 
--is_freezing=False 
--no_freezing_epoch=0 
$ multi-task weights
--P_loss_weights=1 
--E_loss_weights=1

#### stl
* runing script:
python STL_general_train.py 
--dataset=SFEW (support emotion,pose,gender,age datasets)
--model=vggFace 
--epoch=64 
--batch_size=32 
$ triaing tricks
--is_augmentation=False 
--is_dropout=False 
--is_bn=False 
--weights_decay=0 
--is_freezing=False 
--no_freezing_epoch=0 
$ task type
--task_type=0(refer to the definition of model)





