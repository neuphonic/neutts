# Builds espeak-ng for neutts using cmake.
#
# This is called automatically by scikit-build-core from pyproject.toml.
cmake_minimum_required(VERSION 3.26)
project(neutts LANGUAGES C CXX)

include(ExternalProject)

# Install location for espeak-ng
set(ESPEAKNG_BUILD_DIR ${CMAKE_BINARY_DIR}/espeak_ng)
set(ESPEAKNG_INSTALL_DIR ${CMAKE_BINARY_DIR}/espeak_ng-install)

ExternalProject_Add(espeak_ng_external
    GIT_REPOSITORY https://github.com/espeak-ng/espeak-ng.git
    GIT_TAG 69bfd1efde0236654ccc117a28d04aef27673a30
    PREFIX ${ESPEAKNG_BUILD_DIR}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${ESPEAKNG_INSTALL_DIR}
        -DBUILD_SHARED_LIBS:BOOL=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DUSE_ASYNC:BOOL=OFF
        -DUSE_MBROLA:BOOL=OFF
        -DUSE_LIBSONIC:BOOL=OFF
        -DUSE_LIBPCAUDIO:BOOL=OFF
        -DUSE_KLATT:BOOL=OFF
        -DUSE_SPEECHPLAYER:BOOL=OFF
        -DEXTRA_cmn:BOOL=ON
        -DEXTRA_ru:BOOL=ON
        # Need to explicitly add ucd include directory for CI
        "-DCMAKE_C_FLAGS=-D_FILE_OFFSET_BITS=64 -I${ESPEAKNG_BUILD_DIR}/src/espeak_ng_external/src/ucd-tools/src/include"
        "-DCMAKE_CXX_FLAGS=-D_FILE_OFFSET_BITS=64 -I${ESPEAKNG_BUILD_DIR}/src/espeak_ng_external/src/ucd-tools/src/include"
    UPDATE_DISCONNECTED TRUE
)

# Bundle the shared library into the neutts package so phonemizer can load it.
# On Unix: the .so/.dylib lands in lib/. On Windows: the .dll lands in bin/.
if(WIN32)
    install(DIRECTORY ${ESPEAKNG_INSTALL_DIR}/bin/
        DESTINATION neutts
        FILES_MATCHING
        PATTERN "espeak-ng*.dll"
    )
else()
    install(DIRECTORY ${ESPEAKNG_INSTALL_DIR}/lib/
        DESTINATION neutts
        FILES_MATCHING
        PATTERN "libespeak-ng*"
    )
endif()

# Bundle espeak-ng-data so phonemizer can find the voice/language data
install(DIRECTORY ${ESPEAKNG_INSTALL_DIR}/share/espeak-ng-data/
    DESTINATION neutts/espeak-ng-data
)
