# Awesome-Optical-Flow
This is a list of awesome articles about optical flow and related work. [Click here to read in full screen.](https://github.com/hzwer/Awesome-Optical-Flow/blob/main/README.md)

## Optical Flow

### Supervised Models
| Time | Paper | Repo |
| -------- | -------- | -------- |
|ICCV21|[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/abs/2104.02409)|[GMA](https://github.com/zacjiang/GMA) ![Github stars](https://img.shields.io/github/stars/zacjiang/GMA)|
|CVPR21|[Learning Optical Flow from a Few Matches](https://arxiv.org/abs/2104.02166)|[SCV](https://github.com/zacjiang/SCV) ![Github stars](https://img.shields.io/github/stars/zacjiang/SCV)|
|ECCV20|[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)|[RAFT](https://github.com/princeton-vl/RAFT) ![Github stars](https://img.shields.io/github/stars/princeton-vl/RAFT)
|CVPR20|[MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask](https://arxiv.org/abs/2003.10955)|[MaskFlownet](https://github.com/microsoft/MaskFlownet) ![Github stars](https://img.shields.io/github/stars/microsoft/MaskFlownet)
|CVPR20|[ScopeFlow: Dynamic Scene Scoping for Optical Flow](https://arxiv.org/abs/2002.10770)|[ScopeFlow](https://github.com/avirambh/ScopeFlow) ![Github stars](https://img.shields.io/github/stars/avirambh/ScopeFlow)
|TPAMI20|[A Lightweight Optical Flow CNN - Revisiting Data Fidelity and Regularization](https://arxiv.org/abs/1903.07414)|[LiteFlowNet2](https://github.com/twhui/LiteFlowNet2) ![Github stars](https://img.shields.io/github/stars/twhui/LiteFlowNet2)
|NeurIPS19|[Volumetric Correspondence Networks for Optical Flow](https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html)|[VCN](https://github.com/gengshan-y/VCN) ![Github stars](https://img.shields.io/github/stars/gengshan-y/VCN)
|CVPR19|[Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation](https://arxiv.org/pdf/1904.05290.pdf)|[irr](https://github.com/visinf/irr) ![Github stars](https://img.shields.io/github/stars/visinf/irr)
|CVPR18|[PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371)|[PWC-Net](https://github.com/NVlabs/PWC-Net) ![Github stars](https://img.shields.io/github/stars/NVlabs/PWC-Net) | [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc) ![Github stars](https://img.shields.io/github/stars/sniklaus/pytorch-pwc) 
|CVPR18|[LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation](https://arxiv.org/abs/1805.07036)|[LiteFlowNet](https://github.com/twhui/LiteFlowNet) ![Github stars](https://img.shields.io/github/stars/twhui/LiteFlowNet) | [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet) ![Github stars](https://img.shields.io/github/stars/sniklaus/pytorch-liteflownet)
|CVPR17|[FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925)|[flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) ![Github stars](https://img.shields.io/github/stars/NVIDIA/flownet2-pytorch) <br> [flownet2](https://github.com/lmb-freiburg/flownet2) ![Github stars](https://img.shields.io/github/stars/lmb-freiburg/flownet2) <br> [flownet2-tf](https://github.com/sampepose/flownet2-tf) ![Github stars](https://img.shields.io/github/stars/sampepose/flownet2-tf)
|CVPR17|[Optical Flow Estimation using a Spatial Pyramid Network](https://arxiv.org/abs/1611.00850)|[spynet](https://github.com/anuragranj/spynet) ![Github stars](https://img.shields.io/github/stars/anuragranj/spynet) | [pytorch-spynet](https://github.com/sniklaus/pytorch-spynet) ![Github stars](https://img.shields.io/github/stars/sniklaus/pytorch-spynet)
|ICCV15|[FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852)|[FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) ![Github stars](https://img.shields.io/github/stars/ClementPinard/FlowNetPytorch)

### Data Synthesis
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR21|[Learning a Better Training Set for Optical Flow](https://arxiv.org/abs/2104.14544)|[Code coming](https://autoflow-google.github.io/#code)
|CVPR21|[Learning Optical Flow from Still Images](https://arxiv.org/abs/2104.03965)|[depthstillation](https://github.com/mattpoggi/depthstillation) ![Github stars](https://img.shields.io/github/stars/mattpoggi/depthstillation)

### Unsupervised Models
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR21|[SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image Warping](https://arxiv.org/abs/2105.07014)
|CVPR21|[UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_UPFlow_Upsampling_Pyramid_for_Unsupervised_Optical_Flow_Learning_CVPR_2021_paper.pdf)|[UPFlow_pytorch](https://github.com/coolbeam/UPFlow_pytorch)
|ECCV20|[What Matters in Unsupervised Optical Flow](https://arxiv.org/abs/2006.04902)|[uflow](https://github.com/google-research/google-research/tree/master/uflow) GoogleResearch
|CVPR20|[Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation](https://arxiv.org/abs/2003.13045)|[ARFlow](https://github.com/lliuz/ARFlow) ![Github stars](https://img.shields.io/github/stars/lliuz/ARFlow)
|CVPR20|[Flow2Stereo: Effective Self-Supervised Learning of Optical Flow and Stereo Matching](https://arxiv.org/abs/2004.02138)
|AAAI19|[DDFlow: Learning Optical Flow with Unlabeled Data Distillation](https://arxiv.org/abs/1902.09145)|[DDFlow](https://github.com/ppliuboy/DDFlow) ![Github stars](https://img.shields.io/github/stars/ppliuboy/DDFlow)
|CVPR19|[SelFlow: Self-Supervised Learning of Optical Flow](https://arxiv.org/abs/1904.09117)|[SelFlow](https://github.com/ppliuboy/SelFlow) ![Github stars](https://img.shields.io/github/stars/ppliuboy/SelFlow)
|CVPR18|[Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose](https://arxiv.org/abs/1803.02276)|[GeoNet](https://github.com/yzcjtr/GeoNet) ![Github stars](https://img.shields.io/github/stars/yzcjtr/GeoNet)

### Classical Methods
| Time | Paper | Repo |
| -------- | -------- | -------- |
|IJCAI1981|[An iterative image registration technique with an application to stereo vision](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=C41563DCDDC44CB0E13D6D64D89FF3FD?doi=10.1.1.421.4619&rep=rep1&type=pdf)||
|AI1981|[Determining optical flow](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.562&rep=rep1&type=pdf)|
|TPAMI10|[Motion Detail Preserving Optical Flow Estimation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.896&rep=rep1&type=pdf)
|CVPR10|[Secrets of Optical Flow Estimation and Their Principles](https://users.soe.ucsc.edu/~pang/200/f18/papers/2018/05539939.pdf)
|ICCV13|[DeepFlow: Large Displacement Optical Flow with Deep Matching](https://openaccess.thecvf.com/content_iccv_2013/papers/Weinzaepfel_DeepFlow_Large_Displacement_2013_ICCV_paper.pdf)|[Project](https://thoth.inrialpes.fr/src/deepflow/)
|ECCV14|[Optical Flow Estimation with Channel Constancy](https://link.springer.com/content/pdf/10.1007/978-3-319-10590-1_28.pdf)
|CVPR17|[S2F: Slow-To-Fast Interpolator Flow](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_S2F_Slow-To-Fast_Interpolator_CVPR_2017_paper.pdf)

### Classical Methods
| Time | Paper | Repo |
| -------- | -------- | -------- |
|IJCAI1981|[An iterative image registration technique with an application to stereo vision](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=C41563DCDDC44CB0E13D6D64D89FF3FD?doi=10.1.1.421.4619&rep=rep1&type=pdf)||
|AI1981|[Determining optical flow](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.562&rep=rep1&type=pdf)|
|TPAMI10|[Motion Detail Preserving Optical Flow Estimation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.896&rep=rep1&type=pdf)
|CVPR10|[Secrets of Optical Flow Estimation and Their Principles](https://users.soe.ucsc.edu/~pang/200/f18/papers/2018/05539939.pdf)
|ICCV13|[DeepFlow: Large Displacement Optical Flow with Deep Matching](https://openaccess.thecvf.com/content_iccv_2013/papers/Weinzaepfel_DeepFlow_Large_Displacement_2013_ICCV_paper.pdf)|[Project](https://thoth.inrialpes.fr/src/deepflow/)
|ECCV14|[Optical Flow Estimation with Channel Constancy](https://link.springer.com/content/pdf/10.1007/978-3-319-10590-1_28.pdf)
|CVPR17|[S2F: Slow-To-Fast Interpolator Flow](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_S2F_Slow-To-Fast_Interpolator_CVPR_2017_paper.pdf)

### Special Scene
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR20|[Optical Flow in Dense Foggy Scenes using Semi-Supervised Learning](https://arxiv.org/abs/2004.01905)
|CVPR20|[Optical Flow in the Dark](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Optical_Flow_in_the_Dark_CVPR_2020_paper.html)|[Optical-Flow-in-the-Dark](https://github.com/mf-zhang/Optical-Flow-in-the-Dark) ![Github stars](https://img.shields.io/github/stars/mf-zhang/Optical-Flow-in-the-Dark)
|ICCV19|[RainFlow: Optical Flow under Rain Streaks and Rain Veiling Effect](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_RainFlow_Optical_Flow_Under_Rain_Streaks_and_Rain_Veiling_Effect_ICCV_2019_paper.html)
|CVPR18|[Robust Optical Flow Estimation in Rainy Scenes](https://arxiv.org/abs/1704.05239)

### Special Device

**Event Camera** [event-based_vision_resources](https://github.com/uzh-rpg/event-based_vision_resources#optical-flow-estimation) ![Github stars](https://img.shields.io/github/stars/uzh-rpg/event-based_vision_resources#optical-flow-estimation)

| Time | Paper | Repo |
| -------- | -------- | -------- |
|ICCV21|[GyroFlow: Gyroscope-Guided Unsupervised Optical Flow Learning](https://arxiv.org/abs/2103.13725)|[GyroFlow](https://github.com/megvii-research/GyroFlow) ![Github stars](https://img.shields.io/github/stars/megvii-research/GyroFlow)
## Scene Flow
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR21|[RAFT-3D: Scene Flow Using Rigid-Motion Embeddings](https://arxiv.org/pdf/2012.00726.pdf)
|CVPR21|[Just Go With the Flow: Self-Supervised Scene Flow Estimation](https://arxiv.org/pdf/1912.00497.pdf)|[Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation](https://github.com/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation) ![Github stars](https://img.shields.io/github/stars/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation)
|CVPR20|[Upgrading Optical Flow to 3D Scene Flow through Optical Expansion](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Upgrading_Optical_Flow_to_3D_Scene_Flow_Through_Optical_Expansion_CVPR_2020_paper.html)|[expansion](https://github.com/gengshan-y/expansion) ![Github stars](https://img.shields.io/github/stars/gengshan-y/expansion)
|CVPR20|[Self-Supervised Monocular Scene Flow Estimation](https://arxiv.org/abs/2004.04143)|[self-mono-sf](https://github.com/visinf/self-mono-sf) ![Github stars](https://img.shields.io/github/stars/visinf/self-mono-sf)

## Applications
### Video Frame Interpolation
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR20|[Softmax Splatting for Video Frame Interpolation](https://arxiv.org/abs/2003.05534)|[softmax-splatting](https://github.com/sniklaus/softmax-splatting) ![Github stars](https://img.shields.io/github/stars/sniklaus/softmax-splatting)
|CVPR20|[Adaptive Collaboration of Flows for Video Frame Interpolation](https://arxiv.org/abs/1907.10244)|[AdaCoF-pytorch](https://github.com/HyeongminLEE/AdaCoF-pytorch) ![Github stars](https://img.shields.io/github/stars/HyeongminLEE/AdaCoF-pytorch)
|CVPR20|[FeatureFlow: Robust Video Interpolation via Structure-to-Texture Generation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gui_FeatureFlow_Robust_Video_Interpolation_via_Structure-to-Texture_Generation_CVPR_2020_paper.pdf)|[FeatureFlow](https://github.com/CM-BF/FeatureFlow) ![Github stars](https://img.shields.io/github/stars/CM-BF/FeatureFlow)
|NIPS19|[Quadratic Video Interpolation](https://arxiv.org/abs/1911.00627)
|CVPR19|[Depth-Aware Video Frame Interpolation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bao_Depth-Aware_Video_Frame_Interpolation_CVPR_2019_paper.pdf)|[DAIN](https://github.com/baowenbo/DAIN) ![Github stars](https://img.shields.io/github/stars/baowenbo/DAIN)
|CVPR18|[Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation](https://arxiv.org/abs/1712.00080)|[Super-SloMo](https://github.com/avinashpaliwal/Super-SloMo) ![Github stars](https://img.shields.io/github/stars/avinashpaliwal/Super-SloMo)
|ICCV17|[Video Frame Synthesis using Deep Voxel Flow](https://arxiv.org/abs/1702.02463)|[voxel-flow](https://github.com/liuziwei7/voxel-flow) ![Github stars](https://img.shields.io/github/stars/liuziwei7/voxel-flow) | [pytorch-voxel-flow](https://github.com/lxx1991/pytorch-voxel-flow) ![Github stars](https://img.shields.io/github/stars/lxx1991/pytorch-voxel-flow)

### Video Action Recognition
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR18|[Optical Flow Guided Feature: A Fast and Robust Motion Representation for Video Action Recognition](https://arxiv.org/abs/1711.11152)|[Optical-Flow-Guided-Feature](https://github.com/kevin-ssy/Optical-Flow-Guided-Feature) ![Github stars](https://img.shields.io/github/stars/kevin-ssy/Optical-Flow-Guided-Feature)
|GCPR18|[On the Integration of Optical Flow and Action Recognition](https://arxiv.org/abs/1712.08416)

### Video Object Segmentation
| Time | Paper | Repo |
| -------- | -------- | -------- |
|ICCV17|[SegFlow: Joint Learning for Video Object Segmentation and Optical Flow](https://arxiv.org/abs/1709.06750)|[SegFlow](https://github.com/JingchunCheng/SegFlow) ![Github stars](https://img.shields.io/github/stars/JingchunCheng/SegFlow)

### Video Stabilization
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR20|[Learning Video Stabilization Using Optical Flow](https://cseweb.ucsd.edu/~ravir/jiyang_cvpr20.pdf)|[jiyang.fun](https://drive.google.com/file/d/1wQJYFd8TMbCRzhmFfDyBj7oHAGfyr1j6/view)
|CVPR14|[Spatially Smooth Optical Flow for Video Stabilization](http://www.liushuaicheng.org/CVPR2014/SteadyFlow.pdf)

### Low Level Vision
| Time | Paper | Repo |
| -------- | -------- | -------- |
|ICCV21|[Deep Reparametrization of Multi-Frame Super-Resolution and Denoising](https://arxiv.org/abs/2108.08286)|[deep-rep](https://github.com/goutamgmb/deep-rep) ![Github stars](https://img.shields.io/github/stars/goutamgmb/deep-rep)
|CVPR21|[Deep Burst Super-Resolution](https://arxiv.org/abs/2101.10997)|[deep-burst-sr](https://github.com/goutamgmb/deep-burst-sr) ![Github stars](https://img.shields.io/github/stars/goutamgmb/deep-burst-sr)
|TIP20|[Deep video super-resolution using HR optical flow estimation](https://arxiv.org/abs/2001.02129)|[SOF-VSR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/SOF-VSR) ![Github stars](https://img.shields.io/github/stars/The-Learning-And-Vision-Atelier-LAVA/SOF-VSR)
