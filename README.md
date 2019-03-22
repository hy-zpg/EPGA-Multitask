## data:
   * storing datasets
   * prepare images-labels pairs
   * generating csv file

## utils:
   stl: prepare data generator for single-task learning
   confusion: prepare data generator for multi-task learning with joint training method
   alternative: prepare data generator for single-task learning with alternative triaing method

## training scripts for different methods
### alternative-mtl:
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

### confusion-mtl
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

### stl
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





