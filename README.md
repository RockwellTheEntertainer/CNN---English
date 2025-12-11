# Introduction

At the beginning of 2022, a simple CNN convolutional neural network training demo was written. The current version is only used to verify the forward and back propagation of the neural network and deepen personal understanding. It does not involve high-performance optimization and model deployment for the time being. It currently only writes a simple implementation on the CPU. There are plans to launch a CPU optimized version and a GPU version (based on CUDA) in the future.

Some basic layers are implemented

- Conv2d Convolutional layer, padding not supported 
- Maximum pooling layer
- ReLU layer
- Linear fully connected layer
- Batch Normalization layer (performance was poor during validation, not yet resolved)
- Dropout (Performance was very poor during verification and has not been resolved yet)

There is also a training process

- Cross entropy
- Stochastic Gradient Descent

There are also definitions of tensor Tensor structures, visualization with the help of gradCAM principles, etc.

# Environment

- Windows 10
- GCC 10.3.0（[TDM-GCC](https://link.zhihu.com/?target=https%3A//jmeubank.github.io/tdm-gcc/download/)）、C++17
- CMake 3.17
- OpenCV 4.5
- XMake  2.7.4（optional）
- [Dataset](https://github.com/hermosayhl/CNN/tree/main/datasets) From [cat-dog-panda](https://link.zhihu.com/?target=https%3A//www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda) dataset calling cat（cat and dog Classification is relatively difficult），then from [CUB-200 bird](https://link.zhihu.com/?target=http%3A//www.vision.caltech.edu/visipedia/CUB-200.html) 1,000 bird images were randomly drawn from the dataset to form a small dataset of three categories.。train : valid : test proportion 8:1:1。

【Note】：

- gcc version >= 10, the code contains the contents of C++17 std::filesystem；
- CMake latest version；
- OpenCV OpenCV is best compiled on Windows based on gcc to ensure smooth linking
If MSVC is the compiler, you can compile one, or you can directly download the official compiled [OpenCV] (https://sourceforge.net/projects/opencvlibrary/files/4.5.5/opencv-4.5.5-vc14_vc15.exe/download)。如果在 On Windows, select gcc as the compiler, and generally compile one from scratch. The following is a simple compilation process with many options turned off (if ffmpeg cannot be downloaded, you can also set WITH_FFMPEG to OFF).

```bash
mkdir build
cd build
# Compilation Options
cmake .. -G "MinGW Makefiles"  -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=./install -D ENABLE_FAST_MATH=ON -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D BUILD_opencv_python_tests=OFF -D BUILD_opencv_python_bindings_generator=OFF -D BUILD_JAVA=OFF -D BUILD_opencv_java_bindings_generator=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_world=OFF -D OPENCV_ENABLE_NONFREE=OFF -D OPENCV_GENERATE_SETUPVARS=OFF -D WITH_OPENCL_D3D11_NV=OFF -D WITH_MSMF=OFF -D WITH_CUDA=OFF -D WITH_FFMPEG=OFF
# Build
mingw32-make -j4
mingw32-make install
# Add the build/install/x64/bin path to the environment variable to ensure it is linked correctly
```

# Start

## **Cmake**

Enter the [cpu](https://github.com/hermosayhl/CNN/tree/main/cpu) directory, create the build directory, generate directory

```bash
mkdir build_cmake
mkdir bin
cd build
```

Execute CMake to generate Makefiles

```bash
cmake .. -G "MinGW Makefiles"
```

I added '-G "MinGW Makefiles"' because I have MSVC on my computer and MSVC is automatically preferred, but I prefer GCC, so I need to add this one. If I'm on Linux, I can leave it out.

Compile Build

```bash
# Windows + GCC
mingw32-make -j4
# Linux + GCC
make -j4
```

I executed the previous command and can see the three generated files in the bin directory

![image-20230208212620325](./imgs/image-20230208212620325.png)

Click train.exe to train the model

![image-20230208213513653](imgs/train.gif)

Click reference.exe to load the trained model and perform inference on a single image

![image-20230208213627060](imgs/image-20230208213627060.png)

Click gradCAM.exe to try the CNN visualization

![image-20230208213713076](imgs/image-20230208213713076.png)



## Xmake

[Xmake](https://github.com/xmake-io/xmake) is a very convenient building tool that I have recently come into contact with. It is also very fast and modular. I recommend it.

Create the build directory first

```bash
mkdir build
```

Specify the mingw path (note the modification), which can be easily detected as Anaconda's mingw

```
xmake g --mingw=F:\liuchang\environments\TDM-GCC
```

Using the mingw toolchain

```bash
xmake f -p mingw
```

Build all goals, then

```
xmake build
```

Build a specific goal within it`cnn_train`

```bash
xmake build cnn_train
```

Run Target `cnn_train`

```bash
xmake run cnn_train
```

![](imgs/xmake.gif)

Recompile

```bash
xmake -r
```

【MSVC 】If the computer has a visual studio environment, xmake directly without specifying mingw, **after modifying the opencv path**, followed by build and run.

【GCC 】If it is on Linux, there is usually gcc-10 or above, which is also directly xmake, no need to specify, same as MSVC
