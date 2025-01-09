cmake_minimum_required(VERSION 3.25)
project(COSC-4P80-Final-Project VERSION 0.0.26)

option(ENABLE_ADDRSAN "Enable the address sanitizer" OFF)
option(ENABLE_UBSAN "Enable the ub sanitizer" OFF)
option(ENABLE_TSAN "Enable the thread data race sanitizer" OFF)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(lib/blt-with-graphics)

add_compile_options("-fopenmp")
add_link_options("-fopenmp")

fetchcontent_declare(dlib
        URL http://dlib.net/files/dlib-19.24.tar.bz2
        URL_HASH MD5=8a98957a73eebd3cd7431c2bac79665f
        FIND_PACKAGE_ARGS)
fetchcontent_makeavailable(dlib)

include_directories(include/)
file(GLOB_RECURSE PROJECT_BUILD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(COSC-4P80-Final-Project ${PROJECT_BUILD_FILES})

target_compile_options(COSC-4P80-Final-Project PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)
target_link_options(COSC-4P80-Final-Project PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)

target_link_libraries(COSC-4P80-Final-Project PRIVATE BLT_WITH_GRAPHICS dlib)

if (${ENABLE_ADDRSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Final-Project PRIVATE -fsanitize=address)
    target_link_options(COSC-4P80-Final-Project PRIVATE -fsanitize=address)
endif ()

if (${ENABLE_UBSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Final-Project PRIVATE -fsanitize=undefined)
    target_link_options(COSC-4P80-Final-Project PRIVATE -fsanitize=undefined)
endif ()

if (${ENABLE_TSAN} MATCHES ON)
    target_compile_options(COSC-4P80-Final-Project PRIVATE -fsanitize=thread)
    target_link_options(COSC-4P80-Final-Project PRIVATE -fsanitize=thread)
endif ()
