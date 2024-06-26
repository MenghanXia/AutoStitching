## Introduction

This project is a generic framework for globally consistent alignment of images captured from approximately planar 
scenes via topology analysis. Specifically, it can resist the perspective distortion meanwhile preserving the local 
alignment accuracy. To guanrantee the alignment accuracy, global topological relations of images are searched firstly,
and then a global optimization on alignment parameters are performed.

This C++ implemented algorithm is described in  
"[Globally Consistent Alignment for Planar Mosaicking via Topology Analysis](http://menghanxia.github.io/papers/2017_Planar_Alignment_pr.pdf)", Pattern Recognition (PR), Jan. 2017.  
Notice: This program is free for personal, non-profit and academic use.
All right reserved to CVRS: http://cvrs.whu.edu.cn/. 
If you have any question, please contact: menghanxyz@gmail.com (Menghan Xia)

Here is an example for demonstration below (image topological graph and alignment result): 

<img src="docs/demo.png" width="900px"/>

## Usage
### Dependent Libarary [compulsory]:
OpenCV 2.4.9 is recommended.

### 1. Project Configure:
This procedure is developed on *Visual Studio 2010* under *Windows 8.1* system environment,
where the source code is organized with CMakeLists. So, before opening it in Visual Studio,
you need to configure the project with *CMake*.

### 2. Running and Test:
2.1 Default folders in "*data*":  
- "*Images*" : put your source images in it.
- "*Cache*" (optional) : used to store those intermediate results (feature points files, matching point files, topological matrix, etc) that 
are required by the final alignmnet optimization.

2.2 Running parameter settings:  
- Set your reference image for alignment: 【Variable '*refNo*'】 [line 16] in [main.cpp](./source/Src/Mosaic/main.cpp). 
e.g. refNo=-1 means that the program will automatically selects a reference via topological analysis.
- Set whether your image set is sequential order or not: 【Variable '*isSeqData*'】 [line 38] in [main.cpp](./source/Src/Mosaic/main.cpp)
- Set whether the image keypoints available in [*Cache*](./data/Cache): 【Variable '*loadKeyPts*'】 [line 39] in [main.cpp](./source/Src/Mosaic/main.cpp)
- Set whether the inter-image overlap relations available in [*Cache*](./data/Cache): 【Variable '*loadTopology*'】 [line 40] in [main.cpp](./source/Src/Mosaic/main.cpp)

Besides, to use our preset working directory successfully, do not forget to UPDATE the path variable 
'*baseDir*' as **the absolute path** of your directory "*data*" in the source file [source/Src/Utils/util.h](./source/Src/Utils/util.h) [line 16]

So far, you can run the procedure and check the alignment/stitching results now.

## Citation
If any part of our paper and code is helpful to your work, please generously cite with:
```
@article{DBLP:journals/pr/XiaYXLZ17,
  author    = {Menghan Xia and Jian Yao and Renping Xie and Li Li and Wei Zhang},
  title     = {Globally consistent alignment for planar mosaicking via topology analysis},
  journal   = {Pattern Recognit.},
  volume    = {66},
  pages     = {239--252},
  year      = {2017}
}
```
