# Build: cmake -S tools/drburst -B BUILD -DDynamoRIO_DIR=$D/cmake -DLLAMA=$W/llama.cpp \
#          -DDRLIBS=$W/mamba/envs/drlibs
#        cmake --build BUILD
# DRLIBS is a conda env with lz4-c, snappy and zlib, which drmemtrace_static needs and
# the DynamoRIO release does not bundle.
cmake_minimum_required(VERSION 3.14)
project(drburst C CXX)
find_package(DynamoRIO REQUIRED)
set(LLAMA "" CACHE PATH "llama.cpp checkout with build-fast/")
set(DRLIBS "" CACHE PATH "prefix with liblz4, libsnappy, libz")
option(ATTR "build llama-drattr (tensor attribution) instead of llama-drburst" OFF)
if (ATTR)
  set(TGT llama-drattr)
  set(SRC drattr_main.cpp)
else ()
  set(TGT llama-drburst)
  set(SRC drburst_main.cpp)
endif ()
set(LLAMA_BIN ${LLAMA}/build-fast/bin)

# The release's imported targets name its build machine's libpthread and libz by
# absolute path (/usr/lib/x86_64-linux-gnu/...), which does not exist here.
get_property(_dr_targets DIRECTORY PROPERTY IMPORTED_TARGETS)
foreach (t ${_dr_targets})
  get_target_property(libs ${t} INTERFACE_LINK_LIBRARIES)
  if (libs)
    string(REPLACE "/usr/lib/x86_64-linux-gnu/libpthread.so" "pthread" libs "${libs}")
    string(REPLACE "/usr/lib/x86_64-linux-gnu/libz.so" "${DRLIBS}/lib/libz.so" libs "${libs}")
    string(REPLACE "/usr/lib/x86_64-linux-gnu/libz.a" "${DRLIBS}/lib/libz.a" libs "${libs}")
    set_target_properties(${t} PROPERTIES INTERFACE_LINK_LIBRARIES "${libs}")
  endif ()
endforeach ()

add_executable(${TGT} ${SRC})
target_include_directories(${TGT} PRIVATE ${LLAMA}/include ${LLAMA}/ggml/include)
configure_DynamoRIO_static(${TGT})
# drmemtrace_static is not an imported target in the release: link it from its
# directory, with the extensions it uses, as DynamoRIO's own burst_static test does.
target_link_directories(${TGT} PRIVATE ${DynamoRIO_DIR}/../tools/lib64/release ${DRLIBS}/lib)
use_DynamoRIO_static_client(${TGT} drmemtrace_static)
use_DynamoRIO_extension(${TGT} drmemtrace_static)
foreach (ext drmgr_static drreg_static drutil_static drx_static drwrap_static drsyms_static
    drcovlib_static drbbdup_static drstatecmp_static drpttracer_static)
  use_DynamoRIO_extension(${TGT} ${ext})
endforeach ()
target_link_libraries(${TGT} ${LLAMA_BIN}/libllama-completion-impl.so ${LLAMA_BIN}/libllama-common.so
  ${LLAMA_BIN}/libllama.so ${LLAMA_BIN}/libggml.so ${LLAMA_BIN}/libggml-base.so ${LLAMA_BIN}/libggml-cpu.so
  ${DRLIBS}/lib/liblz4.so ${DRLIBS}/lib/libsnappy.so ${DRLIBS}/lib/libz.so dl pthread)
# Export llama_process so llama's libraries bind to this one, and find build-fast's
# libraries and the compression libraries at run time.
target_link_options(${TGT} PRIVATE -Wl,-rpath-link,${LLAMA_BIN} -Wl,-rpath,${LLAMA_BIN}:${DRLIBS}/lib)
set_target_properties(${TGT} PROPERTIES ENABLE_EXPORTS ON BUILD_RPATH "${LLAMA_BIN};${DRLIBS}/lib")
