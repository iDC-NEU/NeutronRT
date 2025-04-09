import os
from torch.utils.cpp_extension import load

dir_path = os.path.dirname(os.path.realpath(__file__))

cuda_path = os.environ["CUDA_HOME"]
cuda_include = os.path.join(cuda_path, 'include')
cuda_lib = os.path.join(cuda_path, 'lib64')

pmaGraph = load(name='pmaGraph', sources=[os.path.join(dir_path, 'pma_graph.cu')], 
               extra_cflags=['-g', '-lrt', 
                             '-I', cuda_include, '-L', cuda_lib], 
               extra_ldflags=['-lcuda'],
               extra_cuda_cflags=['-g', '-G', 
                                  '-gencode=arch=compute_70,code=sm_70', 
                                  '-gencode=arch=compute_80,code=sm_80', 
                                  '-gencode=arch=compute_86,code=sm_86',
                                  '-I', cuda_include, '-L', cuda_lib,
                                #   '-ccbin=/usr/bin/gcc-10',
                                  '--extended-lambda',])

gpuDataCache = load(name='gpuDataCache', sources=[os.path.join(dir_path, 'gpu_cache.cu')], 
               extra_cflags=['-g', '-lrt', 
                             '-I', cuda_include, '-L', cuda_lib], 
               extra_ldflags=['-lcuda'], 
               extra_cuda_cflags=['-g', '-G', 
                                  '-gencode=arch=compute_70,code=sm_70', 
                                  '-gencode=arch=compute_80,code=sm_80', 
                                  '-gencode=arch=compute_86,code=sm_86',
                                  '-I', cuda_include, '-L', cuda_lib,
                                #   '-ccbin=/usr/bin/gcc-10',
                                  '--extended-lambda',])