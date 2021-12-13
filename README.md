# Awesome-Optical-Flow
This is a list of awesome articles about optical flow and related work.

## Optical Flow
### Classical Methods
[IJCAI1981 (Lucas-Kanade method) An iterative image registration technique with an application to stereo vision](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=C41563DCDDC44CB0E13D6D64D89FF3FD?doi=10.1.1.421.4619&rep=rep1&type=pdf)

[AI1981 (Horn-Schunck method) Determining optical flow](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.562&rep=rep1&type=pdf)

### Supervised Models
[ICCV21 Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/abs/2104.02409)

Code: [GMA](https://github.com/zacjiang/GMA) ![Github stars](https://img.shields.io/github/stars/zacjiang/GMA)

[CVPR21 Learning Optical Flow from a Few Matches](https://arxiv.org/abs/2104.02166)

Code: [SCV](https://github.com/zacjiang/SCV) ![Github stars](https://img.shields.io/github/stars/zacjiang/SCV)

[ECCV20 RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)

Code: [RAFT](https://github.com/princeton-vl/RAFT) ![Github stars](https://img.shields.io/github/stars/princeton-vl/RAFT)

[CVPR20 MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask](https://arxiv.org/abs/2003.10955) 

Code: [MaskFlownet](https://github.com/microsoft/MaskFlownet) ![Github stars](https://img.shields.io/github/stars/microsoft/MaskFlownet)

[CVPR20 ScopeFlow: Dynamic Scene Scoping for Optical Flow](https://arxiv.org/abs/2002.10770)

Code: [ScopeFlow](https://github.com/avirambh/ScopeFlow) ![Github stars](https://img.shields.io/github/stars/avirambh/ScopeFlow)

[TPAMI20 A Lightweight Optical Flow CNN - Revisiting Data Fidelity and Regularization](https://arxiv.org/abs/1903.07414) 

Code: [LiteFlowNet2](https://github.com/twhui/LiteFlowNet2) ![Github stars](https://img.shields.io/github/stars/twhui/LiteFlowNet2)

[NeurIPS19 Volumetric Correspondence Networks for Optical Flow](https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html)

Code: [VCN](https://github.com/gengshan-y/VCN) ![Github stars](https://img.shields.io/github/stars/gengshan-y/VCN)

[CVPR19 Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation](https://arxiv.org/pdf/1904.05290.pdf) 

Code: [irr](https://github.com/visinf/irr) ![Github stars](https://img.shields.io/github/stars/visinf/irr)

[CVPR18 PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371) 

Code: [PWC-Net](https://github.com/NVlabs/PWC-Net) ![Github stars](https://img.shields.io/github/stars/NVlabs/PWC-Net) | [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc) ![Github stars](https://img.shields.io/github/stars/sniklaus/pytorch-pwc) 

[CVPR18 LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation](https://arxiv.org/abs/1805.07036)

Code: [LiteFlowNet](https://github.com/twhui/LiteFlowNet) ![Github stars](https://img.shields.io/github/stars/twhui/LiteFlowNet) | [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet) ![Github stars](https://img.shields.io/github/stars/sniklaus/pytorch-liteflownet)

[CVPR17 FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925) 

Code: [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) ![Github stars](https://img.shields.io/github/stars/NVIDIA/flownet2-pytorch) | [flownet2](https://github.com/lmb-freiburg/flownet2) ![Github stars](https://img.shields.io/github/stars/lmb-freiburg/flownet2) | [flownet2-tf](https://github.com/sampepose/flownet2-tf) ![Github stars](https://img.shields.io/github/stars/sampepose/flownet2-tf)

[CVPR17 Optical Flow Estimation using a Spatial Pyramid Network](https://arxiv.org/abs/1611.00850)

Code: [spynet](https://github.com/anuragranj/spynet) ![Github stars](https://img.shields.io/github/stars/anuragranj/spynet) | [pytorch-spynet](https://github.com/sniklaus/pytorch-spynet) ![Github stars](https://img.shields.io/github/stars/sniklaus/pytorch-spynet)

[ICCV15 FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852) 

Code: [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) ![Github stars](https://img.shields.io/github/stars/ClementPinard/FlowNetPytorch)

### Data Synthesis
[CVPR21 Learning a Better Training Set for Optical Flow](https://arxiv.org/abs/2104.14544)

[Code coming](https://autoflow-google.github.io/#code)

[CVPR21 Learning Optical Flow from Still Images](https://arxiv.org/abs/2104.03965)

Code: [depthstillation](https://github.com/mattpoggi/depthstillation) ![Github stars](https://img.shields.io/github/stars/mattpoggi/depthstillation)

### Unsupervised Models
[CVPR21 SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image Warping](https://arxiv.org/abs/2105.07014)

[CVPR21 UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_UPFlow_Upsampling_Pyramid_for_Unsupervised_Optical_Flow_Learning_CVPR_2021_paper.pdf)

Code: [UPFlow_pytorch](https://github.com/coolbeam/UPFlow_pytorch)

[ECCV20 What Matters in Unsupervised Optical Flow](https://arxiv.org/abs/2006.04902)

Code: [uflow](https://github.com/google-research/google-research/tree/master/uflow) GoogleResearch

[CVPR20 Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation](https://arxiv.org/abs/2003.13045)

Code: [ARFlow](https://github.com/lliuz/ARFlow) ![Github stars](https://img.shields.io/github/stars/lliuz/ARFlow)

[CVPR20 Flow2Stereo: Effective Self-Supervised Learning of Optical Flow and Stereo Matching](https://arxiv.org/abs/2004.02138)

[AAAI19 DDFlow: Learning Optical Flow with Unlabeled Data Distillation](https://arxiv.org/abs/1902.09145)

Code: [DDFlow](https://github.com/ppliuboy/DDFlow) ![Github stars](https://img.shields.io/github/stars/ppliuboy/DDFlow)

[CVPR19 SelFlow: Self-Supervised Learning of Optical Flow](https://arxiv.org/abs/1904.09117)

Code: [SelFlow](https://github.com/ppliuboy/SelFlow) ![Github stars](https://img.shields.io/github/stars/ppliuboy/SelFlow)

[CVPR18 Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose](https://arxiv.org/abs/1803.02276)

Code: [GeoNet](https://github.com/yzcjtr/GeoNet) ![Github stars](https://img.shields.io/github/stars/yzcjtr/GeoNet)

### Special Scene
[CVPR20 Optical Flow in Dense Foggy Scenes using Semi-Supervised Learning](https://arxiv.org/abs/2004.01905)

[CVPR20 Optical Flow in the Dark](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Optical_Flow_in_the_Dark_CVPR_2020_paper.html)

Code: [Optical-Flow-in-the-Dark](https://github.com/mf-zhang/Optical-Flow-in-the-Dark) ![Github stars](https://img.shields.io/github/stars/mf-zhang/Optical-Flow-in-the-Dark)

[CVPR18 Robust Optical Flow Estimation in Rainy Scenes](https://arxiv.org/abs/1704.05239)

### Special Device

**Event Camera** [event-based_vision_resources](https://github.com/uzh-rpg/event-based_vision_resources#optical-flow-estimation) ![Github stars](https://img.shields.io/github/stars/uzh-rpg/event-based_vision_resources#optical-flow-estimation)

[ICCV21 GyroFlow: Gyroscope-Guided Unsupervised Optical Flow Learning](https://arxiv.org/abs/2103.13725)

Code: [GyroFlow](https://github.com/megvii-research/GyroFlow) ![Github stars](https://img.shields.io/github/stars/megvii-research/GyroFlow)
## Scene Flow
[CVPR21 RAFT-3D: Scene Flow Using Rigid-Motion Embeddings](https://arxiv.org/pdf/2012.00726.pdf)

[CVPR21 Just Go With the Flow: Self-Supervised Scene Flow Estimation](https://arxiv.org/pdf/1912.00497.pdf)

Code: [Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation](https://github.com/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation) ![Github stars](https://img.shields.io/github/stars/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation)

[CVPR20 Upgrading Optical Flow to 3D Scene Flow through Optical Expansion](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Upgrading_Optical_Flow_to_3D_Scene_Flow_Through_Optical_Expansion_CVPR_2020_paper.html)

Code: [expansion](https://github.com/gengshan-y/expansion) ![Github stars](https://img.shields.io/github/stars/gengshan-y/expansion)

[CVPR20 Self-Supervised Monocular Scene Flow Estimation](https://arxiv.org/abs/2004.04143)

Code: [self-mono-sf](https://github.com/visinf/self-mono-sf) ![Github stars](https://img.shields.io/github/stars/visinf/self-mono-sf)

## Applications
### Video Frame Interpolation
[CVPR20 Softmax Splatting for Video Frame Interpolation](https://arxiv.org/abs/2003.05534)

Code: [softmax-splatting](https://github.com/sniklaus/softmax-splatting) ![Github stars](https://img.shields.io/github/stars/sniklaus/softmax-splatting)

[CVPR20 Adaptive Collaboration of Flows for Video Frame Interpolation](https://arxiv.org/abs/1907.10244)

Code: [AdaCoF-pytorch](https://github.com/HyeongminLEE/AdaCoF-pytorch) ![Github stars](https://img.shields.io/github/stars/HyeongminLEE/AdaCoF-pytorch)

[CVPR20 FeatureFlow: Robust Video Interpolation via Structure-to-Texture Generation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gui_FeatureFlow_Robust_Video_Interpolation_via_Structure-to-Texture_Generation_CVPR_2020_paper.pdf)

Code: [FeatureFlow](https://github.com/CM-BF/FeatureFlow) ![Github stars](https://img.shields.io/github/stars/CM-BF/FeatureFlow)

[NIPS19 Quadratic Video Interpolation](https://arxiv.org/abs/1911.00627)

[CVPR19 Depth-Aware Video Frame Interpolation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bao_Depth-Aware_Video_Frame_Interpolation_CVPR_2019_paper.pdf)

Code: [DAIN](https://github.com/baowenbo/DAIN) ![Github stars](https://img.shields.io/github/stars/baowenbo/DAIN)

[CVPR18 Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation](https://arxiv.org/abs/1712.00080)

Code: [Super-SloMo](https://github.com/avinashpaliwal/Super-SloMo) ![Github stars](https://img.shields.io/github/stars/avinashpaliwal/Super-SloMo)

[ICCV17 Video Frame Synthesis using Deep Voxel Flow](https://arxiv.org/abs/1702.02463)

Code: [voxel-flow](https://github.com/liuziwei7/voxel-flow) ![Github stars](https://img.shields.io/github/stars/liuziwei7/voxel-flow) | [pytorch-voxel-flow](https://github.com/lxx1991/pytorch-voxel-flow) ![Github stars](https://img.shields.io/github/stars/lxx1991/pytorch-voxel-flow)

### Video Action Recognition

[CVPR18 Optical Flow Guided Feature: A Fast and Robust Motion Representation for Video Action Recognition](https://arxiv.org/abs/1711.11152)

Code: [Optical-Flow-Guided-Feature](https://github.com/kevin-ssy/Optical-Flow-Guided-Feature) ![Github stars](https://img.shields.io/github/stars/kevin-ssy/Optical-Flow-Guided-Feature)

[GCPR18 On the Integration of Optical Flow and Action Recognition](https://arxiv.org/abs/1712.08416)

### Video Object Segmentation

[ICCV17 SegFlow: Joint Learning for Video Object Segmentation and Optical Flow](https://arxiv.org/abs/1709.06750)

Code: [SegFlow](https://github.com/JingchunCheng/SegFlow) ![Github stars](https://img.shields.io/github/stars/JingchunCheng/SegFlow)

### Video Stabilization
[CVPR20 Learning Video Stabilization Using Optical Flow](https://cseweb.ucsd.edu/~ravir/jiyang_cvpr20.pdf)

Code: [jiyang.fun](https://drive.google.com/file/d/1wQJYFd8TMbCRzhmFfDyBj7oHAGfyr1j6/view)

[CVPR14 Spatially Smooth Optical Flow for Video Stabilization](http://www.liushuaicheng.org/CVPR2014/SteadyFlow.pdf)

### Low Level Vision
[ICCV21 Deep Reparametrization of Multi-Frame Super-Resolution and Denoising](https://arxiv.org/abs/2108.08286)

Code: [deep-rep](https://github.com/goutamgmb/deep-rep) ![Github stars](https://img.shields.io/github/stars/goutamgmb/deep-rep)

[CVPR21 Deep Burst Super-Resolution](https://arxiv.org/abs/2101.10997)

Code: [deep-burst-sr](https://github.com/goutamgmb/deep-burst-sr) ![Github stars](https://img.shields.io/github/stars/goutamgmb/deep-burst-sr)

[TIP20 Deep video super-resolution using HR optical flow estimation](https://arxiv.org/abs/2001.02129)

Code: [SOF-VSR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/SOF-VSR) ![Github stars](https://img.shields.io/github/stars/The-Learning-And-Vision-Atelier-LAVA/SOF-VSR)
