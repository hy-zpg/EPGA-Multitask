There are two kinds of training strategise: joint training and alternative training
1. confusion_MTL is used for joint training to prepare data generator:
   EAP: emotion,age,pose
   EP: emotion,pose
   EGA: emotion,gender,age
   EA: emotion,age
   manifold: generating manifold regularization consists of $l_{1,2}$ norm, nuclear norm and Laplacian norm
2. proposed_MTL is used for alternative training to prepare data generator:
   same as above
3. pseudo_density_distribution is used for generating weights of confidence score, local density and data distribution and selection critiria:
   density_gmm_distribution: density and gmm calculation
   pseudo_label_weights: different selection critiria for pseudo labels
4. two_iput: prepare two set of inputs for multi-task network training
5. STL is used for STL training to prepare data generator:

