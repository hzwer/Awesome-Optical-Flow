# Awesome-Optical-Flow
This is a list of awesome articles about optical flow and related work. [Click here to read in full screen.](https://github.com/hzwer/Awesome-Optical-Flow/blob/main/README.md)

The table of contents is on the left side of the "README.md".

## Optical Flow

### Supervised Models
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR24|[MemFlow: Optical Flow Estimation and Prediction with Memory](https://dqiaole.github.io/MemFlow/)|[MemFlow](https://github.com/DQiaole/MemFlow) ![Github stars](https://img.shields.io/github/stars/DQiaole/MemFlow)|
|CVPR23|[DistractFlow: Improving Optical Flow Estimation via Realistic Distractions and Pseudo-Labeling](https://arxiv.org/abs/2303.14078)
|CVPR23|[Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2023/html/Shi_FlowFormer_Masked_Cost_Volume_Autoencoding_for_Pretraining_Optical_Flow_Estimation_CVPR_2023_paper.html)|[FlowFormerPlusPlus](https://github.com/XiaoyuShi97/FlowFormerPlusPlus) ![Github stars](https://img.shields.io/github/stars/XiaoyuShi97/FlowFormerPlusPlus)|
|NeurIPS22|[SKFlow: Learning Optical Flow with Super Kernels](https://openreview.net/forum?id=v2es9YoukWO)|[SKFlow](https://github.com/littlespray/SKFlow) ![Github stars](https://img.shields.io/github/stars/littlespray/SKFlow)|
|ECCV22|[Disentangling architecture and training for optical flow](https://arxiv.org/abs/2203.10712)|[Autoflow](https://github.com/google-research/opticalflow-autoflow) ![Github stars](https://img.shields.io/github/stars/google-research/opticalflow-autoflow)|
|ECCV22|[FlowFormer: A Transformer Architecture for Optical Flow](https://arxiv.org/pdf/2203.16194.pdf)|[FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official/) ![Github stars](https://img.shields.io/github/stars/drinkingcoder/FlowFormer-Official)|
|CVPR22|[Learning Optical Flow with Kernel Patch Attention](https://openaccess.thecvf.com/content/CVPR2022/papers/Luo_Learning_Optical_Flow_With_Kernel_Patch_Attention_CVPR_2022_paper.pdf)|[KPAFlow](https://github.com/megvii-research/KPAFlow) ![Github stars](https://img.shields.io/github/stars/megvii-research/KPAFlow)|
|CVPR22|[GMFlow: Learning Optical Flow via Global Matching](https://arxiv.org/abs/2111.13680)|[gmflow](https://github.com/haofeixu/gmflow) ![Github stars](https://img.shields.io/github/stars/haofeixu/gmflow)|
|CVPR22|[Deep Equilibrium Optical Flow Estimation](https://arxiv.org/pdf/2204.08442.pdf)|[deq-flow](https://github.com/locuslab/deq-flow) ![Github stars](https://img.shields.io/github/stars/locuslab/deq-flow)|
|ICCV21|[High-Resolution Optical Flow from 1D Attention and Correlation](https://arxiv.org/abs/2104.13918)|[flow1d](https://github.com/haofeixu/flow1d)![Github stars](https://img.shields.io/github/stars/haofeixu/flow1d)|
|ICCV21|[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/abs/2104.02409)|[GMA](https://github.com/zacjiang/GMA) ![Github stars](https://img.shields.io/github/stars/zacjiang/GMA)|
|CVPR21|[Learning Optical Flow from a Few Matches](https://arxiv.org/abs/2104.02166)|[SCV](https://github.com/zacjiang/SCV) ![Github stars](https://img.shields.io/github/stars/zacjiang/SCV)|
|TIP21|[Detail Preserving Coarse-to-Fine Matching for Stereo Matching and Optical Flow](https://ieeexplore.ieee.org/document/9459444)
|ECCV20|[RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://arxiv.org/pdf/2003.12039.pdf)|[RAFT](https://github.com/princeton-vl/RAFT) ![Github stars](https://img.shields.io/github/stars/princeton-vl/RAFT)
|CVPR20|[MaskFlownet: Asymmetric Feature Matching with Learnable Occlusion Mask](https://arxiv.org/abs/2003.10955)|[MaskFlownet](https://github.com/microsoft/MaskFlownet) ![Github stars](https://img.shields.io/github/stars/microsoft/MaskFlownet)
|CVPR20|[ScopeFlow: Dynamic Scene Scoping for Optical Flow](https://arxiv.org/abs/2002.10770)|[ScopeFlow](https://github.com/avirambh/ScopeFlow) ![Github stars](https://img.shields.io/github/stars/avirambh/ScopeFlow)
|TPAMI20|[A Lightweight Optical Flow CNN - Revisiting Data Fidelity and Regularization](https://arxiv.org/abs/1903.07414)|[LiteFlowNet2](https://github.com/twhui/LiteFlowNet2) ![Github stars](https://img.shields.io/github/stars/twhui/LiteFlowNet2)

### Multi-Frame Supervised Models
| Time | Paper | Repo |
| -------- | -------- | -------- |
|ECCV24|[Local All-Pair Correspondence for Point Tracking](https://arxiv.org/abs/2407.15420)
|CVPR24|[FlowTrack: Revisiting Optical Flow for Long-Range Dense Tracking](https://openaccess.thecvf.com/content/CVPR2024/html/Cho_FlowTrack_Revisiting_Optical_Flow_for_Long-Range_Dense_Tracking_CVPR_2024_paper.html)
|CVPR24|[Dense Optical Tracking: Connecting the Dots](https://arxiv.org/abs/2312.00786)|[dot](https://github.com/16lemoing/dot) ![Github stars](https://img.shields.io/github/stars/16lemoing/dot)|
|ICCV23|[Tracking Everything Everywhere All at Once](https://arxiv.org/abs/2306.05422)|[omnimotion](https://github.com/qianqianwang68/omnimotion) ![Github stars](https://img.shields.io/github/stars/qianqianwang68/omnimotion)|
|ICCV23|[AccFlow: Backward Accumulation for Long-Range Optical Flow](https://arxiv.org/pdf/2308.13133.pdf)|[AccFlow](https://github.com/mulns/AccFlow) ![Github stars](https://img.shields.io/github/stars/mulns/AccFlow)|
|ICCV23|[VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation](https://arxiv.org/abs/2303.08340)|[VideoFlow](https://github.com/XiaoyuShi97/VideoFlow) ![Github stars](https://img.shields.io/github/stars/XiaoyuShi97/VideoFlow)|
|ECCV22|[Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories](https://arxiv.org/abs/2204.04153)|[PIPs](https://github.com/aharley/pips) ![Github stars](https://img.shields.io/github/stars/aharley/pips)|


### Semi-Supervised Models
| Time | Paper | Repo |
| -------- | -------- | -------- |
|ECCV22|[Semi-Supervised Learning of Optical Flow by Flow Supervisor](https://arxiv.org/abs/2207.10314)

### Data Synthesis
| Time | Paper | Repo |
| -------- | -------- | -------- |
|ECCV22|[RealFlow: EM-based Realistic Optical Flow Dataset Generation from Videos]()|[RealFlow](https://github.com/megvii-research/RealFlow) ![Github stars](https://img.shields.io/github/stars/megvii-research/RealFlow)
|CVPR21|[AutoFlow: Learning a Better Training Set for Optical Flow](https://arxiv.org/abs/2104.14544)|[autoflow](https://github.com/google-research/opticalflow-autoflow) ![Github stars](https://img.shields.io/github/stars/google-research/opticalflow-autoflow)
|CVPR21|[Learning Optical Flow from Still Images](https://arxiv.org/abs/2104.03965)|[depthstillation](https://github.com/mattpoggi/depthstillation) ![Github stars](https://img.shields.io/github/stars/mattpoggi/depthstillation)
|arXiv21.04|[Optical Flow Dataset Synthesis from Unpaired Images](https://arxiv.org/abs/2104.02615)

### Unsupervised Models
| Time | Paper | Repo |
| -------- | -------- | -------- |
|ECCV22|[Optical Flow Training under Limited Label Budget via Active Learning](https://arxiv.org/pdf/2203.05053.pdf)|[optical-flow-active-learning-release](https://github.com/duke-vision/optical-flow-active-learning-release) ![Github stars](https://img.shields.io/github/stars/duke-vision/optical-flow-active-learning-release)
|CVPR21|[SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image Warping](https://arxiv.org/abs/2105.07014)|[smurf](https://github.com/google-research/google-research/tree/master/smurf) GoogleResearch
|CVPR21|[UPFlow: Upsampling Pyramid for Unsupervised Optical Flow Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Luo_UPFlow_Upsampling_Pyramid_for_Unsupervised_Optical_Flow_Learning_CVPR_2021_paper.pdf)|[UPFlow_pytorch](https://github.com/coolbeam/UPFlow_pytorch) ![Github stars](https://img.shields.io/github/stars/coolbeam/UPFlow_pytorch)
|TIP21|[OccInpFlow: Occlusion-Inpainting Optical Flow Estimation by Unsupervised Learning](https://arxiv.org/abs/2006.16637)|[depthstillation](https://github.com/coolbeam/OIFlow) ![Github stars](https://img.shields.io/github/stars/coolbeam/OIFlow)
|ECCV20|[What Matters in Unsupervised Optical Flow](https://arxiv.org/abs/2006.04902)|[uflow](https://github.com/google-research/google-research/tree/master/uflow) GoogleResearch
|CVPR20|[Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation](https://arxiv.org/abs/2003.13045)|[ARFlow](https://github.com/lliuz/ARFlow) ![Github stars](https://img.shields.io/github/stars/lliuz/ARFlow)
|CVPR20|[Flow2Stereo: Effective Self-Supervised Learning of Optical Flow and Stereo Matching](https://arxiv.org/abs/2004.02138)

### Joint Learning
| Time | Paper | Repo |
| -------- | -------- | -------- |
|arXiv21.11|[Unifying Flow, Stereo and Depth Estimation](https://arxiv.org/abs/2211.05783)|[unimatch](https://github.com/autonomousvision/unimatch) ![Github stars](https://img.shields.io/github/stars/autonomousvision/unimatch)|
|CVPR21|[EffiScene: Efficient Per-Pixel Rigidity Inference for Unsupervised Joint Learning of Optical Flow, Depth, Camera Pose and Motion Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Jiao_EffiScene_Efficient_Per-Pixel_Rigidity_Inference_for_Unsupervised_Joint_Learning_of_CVPR_2021_paper.html)
|CVPR21|[Feature-Level Collaboration: Joint Unsupervised Learning of Optical Flow, Stereo Depth and Camera Motion](https://openaccess.thecvf.com/content/CVPR2021/html/Chi_Feature-Level_Collaboration_Joint_Unsupervised_Learning_of_Optical_Flow_Stereo_Depth_CVPR_2021_paper.html)

### Special Scene
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR23|[Unsupervised Cumulative Domain Adaptation for Foggy Scene Optical Flow](https://arxiv.org/abs/2303.07564) |[UCDA-Flow](https://github.com/hyzhouboy/UCDA-Flow) ![Github stars](https://img.shields.io/github/stars/hyzhouboy/UCDA-Flow)
|ECCV22|[Deep 360∘ Optical Flow Estimation Based on Multi-Projection Fusion](https://arxiv.org/abs/2208.00776)
|AAAI21|[Optical flow estimation from a single motion-blurred image](https://www.aaai.org/AAAI21Papers/AAAI-3339.ArgawD.pdf)|
|CVPR20|[Optical Flow in Dense Foggy Scenes using Semi-Supervised Learning](https://arxiv.org/abs/2004.01905)
|CVPR20|[Optical Flow in the Dark](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Optical_Flow_in_the_Dark_CVPR_2020_paper.html)|[Optical-Flow-in-the-Dark](https://github.com/mf-zhang/Optical-Flow-in-the-Dark) ![Github stars](https://img.shields.io/github/stars/mf-zhang/Optical-Flow-in-the-Dark)

### Special Device

**Event Camera** [event-based_vision_resources](https://github.com/uzh-rpg/event-based_vision_resources#optical-flow-estimation) ![Github stars](https://img.shields.io/github/stars/uzh-rpg/event-based_vision_resources#optical-flow-estimation)

| Time | Paper | Repo |
| -------- | -------- | -------- |
|ArXiv23.03|[Learning Optical Flow from Event Camera with Rendered Dataset](https://arxiv.org/abs/2303.11011)
|ECCV22|[Secrets of Event-Based Optical Flow](https://arxiv.org/abs/2207.10022)|[event_based_optical_flow](https://github.com/tub-rip/event_based_optical_flow) ![Github stars](https://img.shields.io/github/stars/tub-rip/event_based_optical_flow)
|ICCV21|[GyroFlow: Gyroscope-Guided Unsupervised Optical Flow Learning](https://arxiv.org/abs/2103.13725)|[GyroFlow](https://github.com/megvii-research/GyroFlow) ![Github stars](https://img.shields.io/github/stars/megvii-research/GyroFlow)

## Scene Flow
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR21|[RAFT-3D: Scene Flow Using Rigid-Motion Embeddings](https://arxiv.org/pdf/2012.00726.pdf)
|CVPR21|[Just Go With the Flow: Self-Supervised Scene Flow Estimation](https://arxiv.org/pdf/1912.00497.pdf)|[Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation](https://github.com/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation) ![Github stars](https://img.shields.io/github/stars/HimangiM/Just-Go-with-the-Flow-Self-Supervised-Scene-Flow-Estimation)
|CVPR21|[Learning to Segment Rigid Motions from Two Frames](https://arxiv.org/abs/2101.03694)|[rigidmask](https://github.com/gengshan-y/rigidmask)![Github stars](https://img.shields.io/github/stars/gengshan-y/rigidmask)
|CVPR20|[Upgrading Optical Flow to 3D Scene Flow through Optical Expansion](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Upgrading_Optical_Flow_to_3D_Scene_Flow_Through_Optical_Expansion_CVPR_2020_paper.html)|[expansion](https://github.com/gengshan-y/expansion) ![Github stars](https://img.shields.io/github/stars/gengshan-y/expansion)
|CVPR20|[Self-Supervised Monocular Scene Flow Estimation](https://arxiv.org/abs/2004.04143)|[self-mono-sf](https://github.com/visinf/self-mono-sf) ![Github stars](https://img.shields.io/github/stars/visinf/self-mono-sf)

## Applications
### Video Synthesis/Generation
| Time | Paper | Repo |
| -------- | -------- | -------- 
|ECCV24|[Clearer Frames, Anytime: Resolving Velocity Ambiguity in Video Frame Interpolation](https://jianwang-cmu.github.io/23VFI/04908.pdf)|[InterpAny-Clearer](https://github.com/zzh-tech/InterpAny-Clearer) ![Github stars](https://img.shields.io/github/stars/zzh-tech/InterpAny-Clearer)
|arXiv23.11|[MoVideo: Motion-Aware Video Generation with Diffusion Models](https://arxiv.org/abs/2311.11325)
|CVPR24|[FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis](https://arxiv.org/pdf/2312.17681.pdf)
|WACV24|[Scale-Adaptive Feature Aggregation for Efficient Space-Time Video Super-Resolution](https://arxiv.org/abs/2310.17294)|[SAFA](https://github.com/megvii-research/WACV2024-SAFA) ![Github stars](https://img.shields.io/github/stars/megvii-research/WACV2024-SAFA)
|CVPR23|[A Dynamic Multi-Scale Voxel Flow Network for Video Prediction](https://arxiv.org/abs/2303.09875)|[DMVFN](https://github.com/megvii-research/CVPR2023-DMVFN) ![Github stars](https://img.shields.io/github/stars/megvii-research/CVPR2023-DMVFN)
|CVPR23|[Conditional Image-to-Video Generation with Latent Flow Diffusion Models](https://openaccess.thecvf.com/content/CVPR2023/papers/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.pdf)|[LFDM](https://github.com/nihaomiao/CVPR23_LFDM) ![Github stars](https://img.shields.io/github/stars/nihaomiao/CVPR23_LFDM)
|CVPR23|[A Unified Pyramid Recurrent Network for Video Frame Interpolation](https://arxiv.org/abs/2211.03456)|[UPR-Net](https://github.com/srcn-ivl/UPR-Net) ![Github stars](https://img.shields.io/github/stars/srcn-ivl/UPR-Net)
|CVPR23|[Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation](https://arxiv.org/abs/2303.00440)|[EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) ![Github stars](https://img.shields.io/github/stars/MCG-NJU/EMA-VFI)
|WACV23|[Frame Interpolation for Dynamic Scenes with Implicit Flow Encoding](https://openaccess.thecvf.com/content/WACV2023/papers/Figueiredo_Frame_Interpolation_for_Dynamic_Scenes_With_Implicit_Flow_Encoding_WACV_2023_paper.pdf)|[frameintIFE](https://github.com/pedrovfigueiredo/frameintIFE) ![Github stars](https://img.shields.io/github/stars/pedrovfigueiredo/frameintIFE)
|ACMMM22|[Neighbor correspondence matching for flow-based video frame synthesis](https://arxiv.org/abs/2207.06763)|
|ECCV22|[Improving the Perceptual Quality of 2D Animation Interpolation](https://arxiv.org/abs/2011.06294)|[eisai](https://github.com/ShuhongChen/eisai-anime-interpolator) ![Github stars](https://img.shields.io/github/stars/ShuhongChen/eisai-anime-interpolator)
|ECCV22|[Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://arxiv.org/abs/2011.06294)|[RIFE](https://github.com/hzwer/ECCV2022-RIFE) ![Github stars](https://img.shields.io/github/stars/hzwer/ECCV2022-RIFE)
|CVPR22|[VideoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution](https://arxiv.org/pdf/2206.04647.pdf)|[VideoINR](https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution) ![Github stars](https://img.shields.io/github/stars/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution)
|CVPR22|[IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation](https://arxiv.org/pdf/2205.14620.pdf)|[IFRNet](https://github.com/ltkong218/IFRNet) ![Github stars](https://img.shields.io/github/stars/ltkong218/IFRNet)
|TOG21|[Neural Frame Interpolation for Rendered Content](https://dl.acm.org/doi/abs/10.1145/3478513.3480553)
|CVPR21|[Deep Animation Video Interpolation in the Wild](https://arxiv.org/abs/2104.02495)|[AnimeInterp](https://github.com/lisiyao21/AnimeInterp) ![Github stars](https://img.shields.io/github/stars/lisiyao21/AnimeInterp)
|CVPR20|[Softmax Splatting for Video Frame Interpolation](https://arxiv.org/abs/2003.05534)|[softmax-splatting](https://github.com/sniklaus/softmax-splatting) ![Github stars](https://img.shields.io/github/stars/sniklaus/softmax-splatting)
|CVPR20|[Adaptive Collaboration of Flows for Video Frame Interpolation](https://arxiv.org/abs/1907.10244)|[AdaCoF-pytorch](https://github.com/HyeongminLEE/AdaCoF-pytorch) ![Github stars](https://img.shields.io/github/stars/HyeongminLEE/AdaCoF-pytorch)
|CVPR20|[FeatureFlow: Robust Video Interpolation via Structure-to-Texture Generation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gui_FeatureFlow_Robust_Video_Interpolation_via_Structure-to-Texture_Generation_CVPR_2020_paper.pdf)|[FeatureFlow](https://github.com/CM-BF/FeatureFlow) ![Github stars](https://img.shields.io/github/stars/CM-BF/FeatureFlow)

### Video Inpainting
| Time | Paper | Repo |
| -------- | -------- | -------- |
|ECCV22|[Flow-Guided Transformer for Video Inpainting](https://arxiv.org/abs/2208.06768)|[FGT](https://github.com/hitachinsk/FGT) ![Github stars](https://img.shields.io/github/stars/hitachinsk/FGT)
|CVPR22|[Inertia-Guided Flow Completion and Style Fusion for Video Inpainting](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Inertia-Guided_Flow_Completion_and_Style_Fusion_for_Video_Inpainting_CVPR_2022_paper.pdf)|[isvi](https://github.com/hitachinsk/isvi) ![Github stars](https://img.shields.io/github/stars/hitachinsk/isvi)

### Video Stabilization
| Time | Paper | Repo |
| -------- | -------- | -------- |
|CVPR20|[Learning Video Stabilization Using Optical Flow](https://cseweb.ucsd.edu/~ravir/jiyang_cvpr20.pdf)|[jiyang.fun](https://drive.google.com/file/d/1wQJYFd8TMbCRzhmFfDyBj7oHAGfyr1j6/view)

### Low Level Vision
| Time | Paper | Repo |
| -------- | -------- | -------- |
|ICCV21|[Deep Reparametrization of Multi-Frame Super-Resolution and Denoising](https://arxiv.org/abs/2108.08286)|[deep-rep](https://github.com/goutamgmb/deep-rep) ![Github stars](https://img.shields.io/github/stars/goutamgmb/deep-rep)
|CVPR21|[Deep Burst Super-Resolution](https://arxiv.org/abs/2101.10997)|[deep-burst-sr](https://github.com/goutamgmb/deep-burst-sr) ![Github stars](https://img.shields.io/github/stars/goutamgmb/deep-burst-sr)
|CVPR20|[Efficient Dynamic Scene Deblurring Using Spatially Variant Deconvolution Network With Optical Flow Guided Training](https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Efficient_Dynamic_Scene_Deblurring_Using_Spatially_Variant_Deconvolution_Network_With_CVPR_2020_paper.html)|
|TIP20|[Deep video super-resolution using HR optical flow estimation](https://arxiv.org/abs/2001.02129)|[SOF-VSR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/SOF-VSR) ![Github stars](https://img.shields.io/github/stars/The-Learning-And-Vision-Atelier-LAVA/SOF-VSR)

### Stereo and SLAM
| Time | Paper | Repo |
| -------- | -------- | -------- |
|3DV21|[RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching](https://arxiv.org/pdf/2109.07547.pdf)|[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) ![Github stars](https://img.shields.io/github/stars/princeton-vl/RAFT-Stereo)
|CVPR20|[VOLDOR: Visual Odometry From Log-Logistic Dense Optical Flow Residuals](https://openaccess.thecvf.com/content_CVPR_2020/html/Min_VOLDOR_Visual_Odometry_From_Log-Logistic_Dense_Optical_Flow_Residuals_CVPR_2020_paper.html)|[VOLDOR](https://github.com/htkseason/VOLDOR) ![Github stars](https://img.shields.io/github/stars/htkseason/VOLDOR)


## Before 2020

### Classical Estimation Methods
| Time | Paper | Repo |
| -------- | -------- | -------- |
|IJCAI1981|[An iterative image registration technique with an application to stereo vision](http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=C41563DCDDC44CB0E13D6D64D89FF3FD?doi=10.1.1.421.4619&rep=rep1&type=pdf)||
|AI1981|[Determining optical flow](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.66.562&rep=rep1&type=pdf)|
|TPAMI10|[Motion Detail Preserving Optical Flow Estimation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.221.896&rep=rep1&type=pdf)
|CVPR10|[Secrets of Optical Flow Estimation and Their Principles](https://users.soe.ucsc.edu/~pang/200/f18/papers/2018/05539939.pdf)
|ICCV13|[DeepFlow: Large Displacement Optical Flow with Deep Matching](https://openaccess.thecvf.com/content_iccv_2013/papers/Weinzaepfel_DeepFlow_Large_Displacement_2013_ICCV_paper.pdf)|[Project](https://thoth.inrialpes.fr/src/deepflow/)
|ECCV14|[Optical Flow Estimation with Channel Constancy](https://link.springer.com/content/pdf/10.1007/978-3-319-10590-1_28.pdf)
|CVPR17|[S2F: Slow-To-Fast Interpolator Flow](https://openaccess.thecvf.com/content_cvpr_2017/papers/Yang_S2F_Slow-To-Fast_Interpolator_CVPR_2017_paper.pdf)

### Others

| Time | Paper | Repo |
| -------- | -------- | -------- |
|NeurIPS19|[Volumetric Correspondence Networks for Optical Flow](https://papers.nips.cc/paper/2019/hash/bbf94b34eb32268ada57a3be5062fe7d-Abstract.html)|[VCN](https://github.com/gengshan-y/VCN) ![Github stars](https://img.shields.io/github/stars/gengshan-y/VCN)
|CVPR19|[Iterative Residual Refinement for Joint Optical Flow and Occlusion Estimation](https://arxiv.org/pdf/1904.05290.pdf)|[irr](https://github.com/visinf/irr) ![Github stars](https://img.shields.io/github/stars/visinf/irr)
|CVPR18|[PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371)|[PWC-Net](https://github.com/NVlabs/PWC-Net) ![Github stars](https://img.shields.io/github/stars/NVlabs/PWC-Net) | [pytorch-pwc](https://github.com/sniklaus/pytorch-pwc) ![Github stars](https://img.shields.io/github/stars/sniklaus/pytorch-pwc) 
|CVPR18|[LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation](https://arxiv.org/abs/1805.07036)|[LiteFlowNet](https://github.com/twhui/LiteFlowNet) ![Github stars](https://img.shields.io/github/stars/twhui/LiteFlowNet) | [pytorch-liteflownet](https://github.com/sniklaus/pytorch-liteflownet) ![Github stars](https://img.shields.io/github/stars/sniklaus/pytorch-liteflownet)
|CVPR17|[FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925)|[flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) ![Github stars](https://img.shields.io/github/stars/NVIDIA/flownet2-pytorch) <br> [flownet2](https://github.com/lmb-freiburg/flownet2) ![Github stars](https://img.shields.io/github/stars/lmb-freiburg/flownet2) <br> [flownet2-tf](https://github.com/sampepose/flownet2-tf) ![Github stars](https://img.shields.io/github/stars/sampepose/flownet2-tf)
|CVPR17|[Optical Flow Estimation using a Spatial Pyramid Network](https://arxiv.org/abs/1611.00850)|[spynet](https://github.com/anuragranj/spynet) ![Github stars](https://img.shields.io/github/stars/anuragranj/spynet) | [pytorch-spynet](https://github.com/sniklaus/pytorch-spynet) ![Github stars](https://img.shields.io/github/stars/sniklaus/pytorch-spynet)
|ICCV15|[FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852)|[FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) ![Github stars](https://img.shields.io/github/stars/ClementPinard/FlowNetPytorch)
|AAAI19|[DDFlow: Learning Optical Flow with Unlabeled Data Distillation](https://arxiv.org/abs/1902.09145)|[DDFlow](https://github.com/ppliuboy/DDFlow) ![Github stars](https://img.shields.io/github/stars/ppliuboy/DDFlow)
|CVPR19|[SelFlow: Self-Supervised Learning of Optical Flow](https://arxiv.org/abs/1904.09117)|[SelFlow](https://github.com/ppliuboy/SelFlow) ![Github stars](https://img.shields.io/github/stars/ppliuboy/SelFlow)
|CVPR19|[Unsupervised Deep Epipolar Flow for Stationary or Dynamic Scenes](https://arxiv.org/abs/1904.03848)|[EPIFlow](https://github.com/yiranzhong/EPIflow) ![Github stars](https://img.shields.io/github/stars/yiranzhong/EPIflow)
|CVPR18|[Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose](https://arxiv.org/abs/1803.02276)|[GeoNet](https://github.com/yzcjtr/GeoNet) ![Github stars](https://img.shields.io/github/stars/yzcjtr/GeoNet)
|ICCV19|[RainFlow: Optical Flow under Rain Streaks and Rain Veiling Effect](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_RainFlow_Optical_Flow_Under_Rain_Streaks_and_Rain_Veiling_Effect_ICCV_2019_paper.html)
|CVPR18|[Robust Optical Flow Estimation in Rainy Scenes](https://arxiv.org/abs/1704.05239)
|NIPS19|[Quadratic Video Interpolation](https://arxiv.org/abs/1911.00627)
|CVPR19|[Depth-Aware Video Frame Interpolation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Bao_Depth-Aware_Video_Frame_Interpolation_CVPR_2019_paper.pdf)|[DAIN](https://github.com/baowenbo/DAIN) ![Github stars](https://img.shields.io/github/stars/baowenbo/DAIN)
|CVPR18|[Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation](https://arxiv.org/abs/1712.00080)|[Super-SloMo](https://github.com/avinashpaliwal/Super-SloMo) ![Github stars](https://img.shields.io/github/stars/avinashpaliwal/Super-SloMo)
|ICCV17|[Video Frame Synthesis using Deep Voxel Flow](https://arxiv.org/abs/1702.02463)|[voxel-flow](https://github.com/liuziwei7/voxel-flow) ![Github stars](https://img.shields.io/github/stars/liuziwei7/voxel-flow) | [pytorch-voxel-flow](https://github.com/lxx1991/pytorch-voxel-flow) ![Github stars](https://img.shields.io/github/stars/lxx1991/pytorch-voxel-flow)
|CVPR19|[DVC: An End-to-end Deep Video Compression Framework](https://arxiv.org/abs/1812.00101)|[PyTorchVideoCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression) ![Github stars](https://img.shields.io/github/stars/ZhihaoHu/PyTorchVideoCompression)
|ICCV17|[SegFlow: Joint Learning for Video Object Segmentation and Optical Flow](https://arxiv.org/abs/1709.06750)|[SegFlow](https://github.com/JingchunCheng/SegFlow) ![Github stars](https://img.shields.io/github/stars/JingchunCheng/SegFlow)
|CVPR18|[End-to-end Flow Correlation Tracking with Spatial-temporal Attention](https://arxiv.org/abs/1711.01124)
|CVPR18|[Optical Flow Guided Feature: A Fast and Robust Motion Representation for Video Action Recognition](https://arxiv.org/abs/1711.11152)|[Optical-Flow-Guided-Feature](https://github.com/kevin-ssy/Optical-Flow-Guided-Feature) ![Github stars](https://img.shields.io/github/stars/kevin-ssy/Optical-Flow-Guided-Feature)
|GCPR18|[On the Integration of Optical Flow and Action Recognition](https://arxiv.org/abs/1712.08416)
|CVPR14|[Spatially Smooth Optical Flow for Video Stabilization](http://www.liushuaicheng.org/CVPR2014/SteadyFlow.pdf)
