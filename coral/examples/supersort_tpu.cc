/* Copyright 2019-2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example to classify image.
// The input image size must match the input size of the model and be stored as
// RGB pixel array.
// In linux, you may resize and convert an existing image to pixel array like:
//   convert cat.bmp -resize 224x224! cat.rgb

/*  BUILD instructions:
    make DOCKER_IMAGE=debian:buster DOCKER_CPUS="aarch64" DOCKER_TARGETS=examples docker-build

    OR for developping, it is easier to build the docker container and compile inside the container:
    docker run -i --tty -v /home/tom/libcoral/:/workspace --name 'my-coral-container' "coral-edgetpu-debian-buster"
        * -i for interactive 
        * --tty for terminal
        * -v to mount the host directory to the container
        * --name to give a name to the container
        * "coral-edgetpu-debian-buster" is the name of the image, make sure it is in the docker images list
    You can start the container with the following command:
        docker start my-coral-container
    Then you can enter the container with the following command:
        docker exec -it my-coral-container /bin/bash
    Then you can compile the code with the following command:
        make CPU="aarch64" COMPILATION_MODE=opt -C workspace/ examples
*/


#include <cmath>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <semaphore.h>
#include <fcntl.h>
#include <unistd.h>


#include "coral/examples/shared_data.h"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/classification/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"

ABSL_FLAG(std::string, model_path, "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
          "Path to the tflite model.");
ABSL_FLAG(int, device_index, 0, "Edge TPU device index.");

int main(int argc, char* argv[]) {
    absl::ParseCommandLine(argc, argv);
    const char * shared_memory_name = std::string(SHARED_MEMORY_NAME + absl::GetFlag(FLAGS_device_index)).c_str();
    const char * sender_semaphore_name = std::string(SENDER_SEMAPHORE_NAME + absl::GetFlag(FLAGS_device_index)).c_str();
    const char * receiver_semaphore_name = std::string(RECEIVER_SEMAPHORE_NAME + absl::GetFlag(FLAGS_device_index)).c_str();

    // Load the model.
    const auto model = coral::LoadModelOrDie(absl::GetFlag(FLAGS_model_path));
    auto edgetpu_context = coral::ContainsEdgeTpuCustomOp(*model)
                                ? coral::GetEdgeTpuContextOrDie(absl::nullopt, 
                                                                absl::GetFlag(FLAGS_device_index),
                                                                {})
                                : nullptr;
    auto interpreter =
        coral::MakeEdgeTpuInterpreterOrDie(*model, edgetpu_context.get());
    CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    CHECK_EQ(interpreter->inputs().size(), 1);
    const auto* input_tensor = interpreter->input_tensor(0);
    CHECK_EQ(input_tensor->type, kTfLiteUInt8)
        << "Only support uint8 input type.";

    auto input = coral::MutableTensorData<uint8_t>(*input_tensor);

    // Open shared memory
    int shm_fd = shm_open(shared_memory_name, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to create shared memory." << std::endl;
        return 1;
    }

    // Set size of shared memory
    if (ftruncate(shm_fd, SHARED_MEMORY_SIZE) == -1) {
        perror("ftruncate");
        close(shm_fd);
        return 1;
    }

    void* shm_ptr = mmap(0, SHARED_MEMORY_SIZE, PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory." << std::endl;
        return 1;
    }

    SharedMemory* data_ptr = static_cast<SharedMemory*>(shm_ptr);
    // Initialize shared memory content
    std::memset(data_ptr, 0, SHARED_MEMORY_SIZE);

    // Open semaphores
    sem_t* sender_sem = sem_open(sender_semaphore_name, O_CREAT, 0666, 0);
    if (sender_sem == SEM_FAILED) {
        perror("sem_open sender");
        munmap(shm_ptr, SHARED_MEMORY_SIZE);
        close(shm_fd);
        return 1;
    }
    sem_t* receiver_sem = sem_open(receiver_semaphore_name, O_CREAT, 0666, 0);
    if (receiver_sem == SEM_FAILED) {
        perror("sem_open receiver");
        sem_close(sender_sem);
        munmap(shm_ptr, SHARED_MEMORY_SIZE);
        close(shm_fd);
        return 1;
    }

    const auto& tensor = *interpreter->output_tensor(0);
    std::cout << "Output tensor scale: " << tensor.params.scale << std::endl;
    std::cout << "Output tensor zero point: " << tensor.params.zero_point << std::endl;
    //std::cout << "Output tensor data type: " << tensor.type << std::endl;



    while(true){
        // Wait for the semaphore
        sem_wait(sender_sem);

        std::cout << "Received image, sending result.." << std::endl;

        const float scale = input_tensor->params.scale;
        const float zero_point = input_tensor->params.zero_point;
        // Read the image from shared memory
        //std::vector<char> image(TPU_IMG_SIZE);
        auto start = std::chrono::high_resolution_clock::now();
        std::memcpy(input.data(), data_ptr->image, TPU_IMG_SIZE);
        auto start2 = std::chrono::high_resolution_clock::now();
        CHECK_EQ(interpreter->Invoke(), kTfLiteOk);
        auto stop = std::chrono::high_resolution_clock::now();

        auto timediff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start2);
        auto timediff_all = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Total execution (copying + tpu) took: " << timediff_all.count() << "microseconds" << std::endl;
        //std::cout << "TPU processing took: " << timediff.count() << "microseconds" << std::endl;

        for (auto result :
            coral::GetClassificationResults(*interpreter, 0.0f, 1)) {
            //std::cout << "---------------------------" << std::endl;
            //std::cout << "Result id: " <<result.id << std::endl;
            std::cout << "Score: " << result.score << std::endl;

            // std::cout << "Result from sender: " << static_cast<int>(data_ptr->result) << std::endl;
	    data_ptr->result = (float) result.score;
        }
        sem_post(receiver_sem);
    }
    // Cleanup
    munmap(shm_ptr, TPU_IMG_SIZE);
    close(shm_fd);
    sem_close(sender_sem);
    sem_close(receiver_sem);

    sem_unlink(sender_semaphore_name);
    shm_unlink(shared_memory_name);

    return 0;
}
