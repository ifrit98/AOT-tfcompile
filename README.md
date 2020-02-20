### 0. Setup workspace
```{bash}
export HOME=/home/localdrive
cd ~
mkdir tfcompile && cd tfcompile
```
#### 1. Clone new tensorflow source tree
```
git clone https://github.com/tensorflow/tensorflow.git
git checkout r2.1 # master branch just broke bazel (https://github.com/tensorflow/tensorflow/commit/09fe958feebec0405ccac225c94fc130304fc2f4)
```
#### 2. Edit and create test graph # In this case test_graph_tfmatmul.pb
  - 2a. Add bitcast to minimal example graph to check for TFCOMPLEX64 support
    nano tensorflow/tensorflow/compiler/aot/tests/make_test_graphs.py
    ``` 
    # add in import section
      from tensorflow import bitcast, complex64
    # add after line 111 in `tfmatmul()`
      x = bitcast(x, complex64)
      y = bitcast(y, complex64)
    ```
  2b. Run script to generate graph and move to //tensorflow/tensorflow
    python tensorflow/tensorflow/compiler/aot/tests/make_test_graphs.py
    mv test_graph_tfmatmul.pb tensorflow/tensorflow/
  ##### NOTE: this errors out with an `Object was never used` message, but all graphs 
  ##### should have been created properly in cwd or `--out_dir` if specified

#### 3. Prepare protobuf config file
cd tensorflow/tensorflow # must be in tensorflow inner workspace 
nano test_graph_matmul.config.pbtxt 

```{test_graph_matmul.config.pbtxt}
# Each feed is a positional input argument for the generated function.  The order
# of each entry matches the order of each input argument.  Here “x_hold” and “y_hold”
# refer to the names of placeholder nodes defined in the graph.
feed {
  id { node_name: "x_hold" }
  shape {
    dim { size: 2 }
    dim { size: 3 }
  }
}
feed {
  id { node_name: "y_hold" }
  shape {
    dim { size: 3 }
    dim { size: 2 }
  }
}

# Each fetch is a positional output argument for the generated function.  The order
# of each entry matches the order of each output argument.  Here “x_y_prod”
# refers to the name of a matmul node defined in the graph.
fetch {
  id { node_name: "x_y_prod" }
}
```

#### 4. Use tf_library macro to compile subgraph.  Append the following to the BUILD file
(from inside tfcompile/tensorflow/tensorflow)
`nano BUILD`

```
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

# Use the tf_library macro to compile your graph into executable code.
tf_library(
    # name is used to generate the following underlying build rules:
    # <name>           : cc_library packaging the generated header and object files
    # <name>_test      : cc_test containing a simple test and benchmark
    # <name>_benchmark : cc_binary containing a stand-alone benchmark with minimal deps;
    #                    can be run on a mobile device
    name = "test_graph_tfmatmul",
    # cpp_class specifies the name of the generated C++ class, with namespaces allowed.
    # The class will be generated in the given namespace(s), or if no namespaces are
    # given, within the global namespace.
    cpp_class = "foo::bar::MatMulComp",
    # graph is the input GraphDef proto, by default expected in binary format.  To
    # use the text format instead, just use the ‘.pbtxt’ suffix.  A subgraph will be
    # created from this input graph, with feeds as inputs and fetches as outputs.
    # No Placeholder or Variable ops may exist in this subgraph.
    graph = "test_graph_tfmatmul.pb",
    # config is the input Config proto, by default expected in binary format.  To
    # use the text format instead, use the ‘.pbtxt’ suffix.  This is where the
    # feeds and fetches were specified above, in the previous step.
    config = "test_graph_tfmatmul.config.pbtxt",
)
```

```{bash}
bazel build :test_graph_matmul
```
 # ERROR: Complex types not supported.  Can be built successfully without adding bitcast lines.
 - ERROR: /home/localdrive/tfcompile/tensorflow/tensorflow/BUILD:937:1: Executing genrule //tensorflow:gen_test_graph_tfmatmul failed (Aborted): bash failed: error executing command /bin/bash -c ... (remaining 1 argument(s) skipped)
2020-02-20 16:02:47.044505: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA
2020-02-20 16:02:47.044891: F tensorflow/compiler/aot/tfcompile_main.cc:175] Non-OK-status: status status: Unimplemented: Complex types not supported.
	 [[{{node Bitcast}}]]
/bin/bash: line 1:  1874 Aborted                 (core dumped) CUDA_VISIBLE_DEVICES='' bazel-out/host/bin/tensorflow/compiler/aot/tfcompile --graph=tensorflow/test_graph_tfmatmul.pb --config=tensorflow/test_graph_tfmatmul.config.pbtxt --entry_point=__xla_tensorflow__test_graph_tfmatmul --cpp_class=foo::bar::MatMulComp --target_triple=x86_64-pc-linux --out_header=bazel-out/k8-opt/bin/tensorflow/test_graph_tfmatmul.h --out_metadata_object=bazel-out/k8-opt/bin/tensorflow/test_graph_tfmatmul_tfcompile_metadata.o --out_function_object=bazel-out/k8-opt/bin/tensorflow/test_graph_tfmatmul_tfcompile_function.o
Target //tensorflow:test_graph_tfmatmul failed to build
Use --verbose_failures to see the command lines of failed build steps.
INFO: Elapsed time: 0.338s, Critical Path: 0.18s
INFO: 4 processes: 4 local.
FAILED: Build did NOT complete successfully


#### 5. Move generated C++ header test_graph_tfmatmul.h and write C++ code to invoke graph
Move generated header file inside tensorflow source-tree to become visible to the bazel BUILD macro
```
mv ../bazel-bin/tensorflow/test_graph_tfmatmul.h ./
nano my_code.cc  # still inside tensorflow/tensorflow
```

```{C++}
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include <iostream>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/test_graph_tfmatmul.h" // generated

int main(int argc, char** argv) {
  Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());


  foo::bar::MatMulComp matmul;
  matmul.set_thread_pool(&device);

  // Set up args and run the computation.
  const float args[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::copy(args + 0, args + 6, matmul.arg0_data());
  std::copy(args + 6, args + 12, matmul.arg1_data());
  matmul.Run();

  // Check result
  if (matmul.result0(0, 0) == 58) {
    std::cout << "Success" << std::endl;
  } else {
    std::cout << "Failed. Expected value 58 at 0,0. Got:"
              << matmul.result0(0, 0) << std::endl;
  }

  return 0;
}
```


#### 6a. Append cc_binary() rule to BUILD 
 ```
# Example of linking your binary
# Also see //tensorflow/compiler/aot/tests/BUILD
# Append this to end of BUILD file

# The executable code generated by tf_library can then be linked into your code.
cc_binary(
    name = "my_binary",
    srcs = [
        "my_code.cc",  # include test_graph_tfmatmul.h to access the generated header
    ],
    deps = [
        ":test_graph_tfmatmul",  # link in the generated object file
        "//third_party/eigen3",
    ],
    linkopts = [
          "-lpthread",
    ]
)

 ```


#### 6b. Run bazel to build final binary in /bazel-bin
 ```{bash}
 bazel build :my_binary
 ```

 #### 7. Invoke binary to test computation without complex64 values (can't build graph binary with complex datatype)
```{bash}
cd ../bazel-bin/tensorflow
./my_binary # run binary, invoking C++ code to run the graph
'Success'

```