# kgraph-gnn


# install
# had to remove this for M1 OSX conflicts
# rm /opt/homebrew/anaconda3/envs/kgraph-gnn/lib/libtapi.dylib

#  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv     --extra-index-url https://data.pyg.org/whl/torch-2.5.1+cpu.html

# pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-2.5.1+cpu.html

# had to install python 3.11 at the system level
# ls /Library/Frameworks/Python.framework/Versions/
# eventually built pyg-lib locally and installed successfully on MacOS
# had to: git submodule update --init --recursive


# export CMAKE_LIBRARY_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib
# export CC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang
# export CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
#  export DYLD_LIBRARY_PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib:$DYLD_LIBRARY_PATH
# sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer
# find /Applications/Xcode.app -name libtapi.dylib

# verify right python linking
# otool -L /opt/homebrew/anaconda3/envs/kgraph-gnn/lib/python3.11/site-packages/libpyg.so
 