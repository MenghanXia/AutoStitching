## Introduction

This project is a generic framework for globally consistent alignment of images captured from approximately planar 
scenes via topology analysis. Specifically, it can resist the perspective distortion meanwhile preserving the local 
alignment accuracy. To guanrantee the alignment accuracy, global topological relations of images are searched firstly,
and then a global optimization on alignment parameters are performed.

This C++ implemented algorithm is described in  
"[Globally Consistent Alignment for Planar Mosaicking via Topology Analysis](http://menghanxia.github.io/papers/Plane_Alignment-PR2016.pdf)", Pattern Recognition (PR), Jan. 2017.  
Notice: This program is free for personal, non-profit and academic use.
All right reserved to CVRS: http://cvrs.whu.edu.cn/. 
If you have any question, please contact: menghanxyz@gmail.com (Menghan Xia)

Here is an example for demonstration below (image topological graph and alignment result): 

<img src="docs/demo.png" width="900px"/>

## Usage
### Dependent Libarary [compulsory]:
OpenCV 2.4.9 is recommended.

### 1. Project Configure:
This procedure is developed on Visual Studio 2010 under windows8.1 system environment,
where the source code is organized with CMakeLists. So, before opening it in Visual Studio,
you need to configure the project with the software named CMake.

### 2. Running and Test:
2.1 Default folders in 'data':

"Images" : put your source images in it.

"Cache"  : used to store those mederate results (feature points files, matching point files, topological matrix, etc) that 
are required by the final alignmnet optimization.

2.2 Running parameter settings:

[a]. set your reference image for alignment      							-->  Variable 'refNo' in Function 'main' in main.cpp. 

e.g. refNo=-1 means the program selects a reference by topological analysis.

[b]. set whether your image set is sequential order or not         --> Variable 'isInorder' in Function 'imageStitcherbyGroup' in alignment.cpp

[c]. set whether your model need global optimization         		--> Variable 'needRefine' in Function 'imageStitcherbyGroup' in alignment.cpp

Besides, to use our preset "data" directory successfully, do not forget to modify the path variable 
"baseDir" to its absolute path of "Data" in the source file "source/Utils/util.h" [line 16]

So far, you can run the procedure and see the alignment/stitching results now.