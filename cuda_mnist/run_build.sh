user="$USER"
rm -r builds
mkdir -p builds
cd builds
export CMAKE_PREFIX_PATH=/home/$user/ld_libs:/opt/cpp_libs/Protobuf:$CMAKE_PREFIX_PATH:/opt/cpp_libs/libtorch::/opt/cpp_libs/config_reader:/opt/cpp_libs/ld_data:/home/liangdao_hanli/software/TensorRT-8.2.5.1
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/home/$user/ld_libs/ld_net ..
cmake --build .
make -j4
make install
