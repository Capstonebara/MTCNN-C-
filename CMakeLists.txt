cmake_minimum_required(VERSION 3.15)  # Ensure a suitable version of CMake
project(FaceRecognitionPipeline)  # Change this to your project name

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the path to the LibTorch installation
set(CMAKE_PREFIX_PATH "/Users/daoxuanbac/libtorch")  # Change this to your actual LibTorch path
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I")
include_directories(/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include/c++/v1)
# Find the Torch package
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Update with the actual path found via `brew info faiss`
set(FAISS_ROOT "/opt/homebrew/Cellar/faiss/1.9.0")
include_directories("${FAISS_ROOT}/include")
find_library(FAISS_LIB faiss HINTS "${FAISS_ROOT}/lib")

# set(NLOHMANN_JSON_ROOT "/opt/homebrew/Cellar/nlohmann-json/3.11.3")
find_package(nlohmann_json REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp src/Camera.cpp src/detector.cpp src/Facenet.cpp src/onet.cpp src/pnet.cpp src/rnet.cpp)

# if (FAISS_LIB)
#     message(STATUS "Found FAISS library: ${FAISS_LIB}")
#     target_link_libraries(${PROJECT_NAME} PRIVATE ${FAISS_LIB})
# else()
#     message(FATAL_ERROR "FAISS library not found.")
# endif()

# Link the Torch library to your executable
# target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Enable C++17
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Set the include directories for LibTorch
target_include_directories(${PROJECT_NAME} PRIVATE "${TORCH_INCLUDE_DIRS}")

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES} ${FAISS_LIB} nlohmann_json::nlohmann_json)

