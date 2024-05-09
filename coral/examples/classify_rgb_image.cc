/* Copyright 2019-2021 Google LLC

SPAGHETTI

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
#include <cmath>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "coral/classification/adapter.h"
#include "coral/examples/file_utils.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/interpreter.h"

#include "coral/tflite_utils.h"
#include <chrono>



ABSL_FLAG(std::string, model_path, "mobilenet_v1_1.0_224_quant_edgetpu.tflite",
          "Path to the tflite model.");
ABSL_FLAG(std::string, image_path, "cat.rgb",
          "Path to the image to be classified. The input image size must match "
          "the input size of the model and the image must be stored as RGB "
          "pixel array.");
ABSL_FLAG(std::string, labels_path, "imagenet_labels.txt",
          "Path to the imagenet labels.");
ABSL_FLAG(float, input_mean, 128, "Mean value for input normalization.");
ABSL_FLAG(float, input_std, 128, "STD value for input normalization.");

int main(int argc, char *argv[])
{
    absl::ParseCommandLine(argc, argv);

    // Load the model.
    const auto model = coral::LoadModelOrDie(absl::GetFlag(FLAGS_model_path));
    auto edgetpu_context = coral::ContainsEdgeTpuCustomOp(*model)
                               ? coral::GetEdgeTpuContextOrDie()
                               : nullptr;
    auto interpreter =
        coral::MakeEdgeTpuInterpreterOrDie(*model, edgetpu_context.get());
    CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    // Check whether input data need to be preprocessed.
    // Image data must go through two transforms before running inference:
    // 1. normalization, f = (v - mean) / std
    // 2. quantization, q = f / scale + zero_point
    // Preprocessing combines the two steps:
    // q = (f - mean) / (std * scale) + zero_point
    // When std * scale equals 1, and mean - zero_point equals 0, the image data
    // do not need any preprocessing. In practice, it is probably okay to skip
    // preprocessing for better efficiency when the normalization and quantization
    // parameters approximate, but do not exactly meet the above conditions.
    CHECK_EQ(interpreter->inputs().size(), 1);
    const auto *input_tensor = interpreter->input_tensor(0);
    CHECK_EQ(input_tensor->type, kTfLiteUInt8)
        << "Only support uint8 input type.";


    // Read image 
    auto input = coral::MutableTensorData<uint8_t>(*input_tensor);
    coral::ReadFileToOrDie(absl::GetFlag(FLAGS_image_path),
                               reinterpret_cast<char *>(input.data()), input.size());

    for (int i=0; i < 100; i++){
        auto start = std::chrono::high_resolution_clock::now();

        CHECK_EQ(interpreter->Invoke(), kTfLiteOk);
        auto stop = std::chrono::high_resolution_clock::now();
        
        
        auto timediff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
        // To get the value of duration use the count()
        // member function on the duration object
        std::cout << "execution took: " << timediff.count() << "microseconds" << std::endl;
    }
    const auto &tensor = interpreter->output_tensor(0);
    std::cout << " Output tensor dtype: " << tensor->type << std::endl;
    
    // Print tensor outputs
    if (tensor->type == kTfLiteUInt8) {
        std::cout << "output is uint8 dtype" << std::endl;
        // const uint8_t* output_data = coral::MutableTensorData<uint8_t>(*tensor);
        // int output_size = coral::GetTensorSize(*tensor);
        // for (int i = 0; i < output_size; ++i) {
        //     std::cout << static_cast<int>(output_data[i]) << " ";
        // }
        absl::Span<const uint8_t> result = coral::TensorData<uint8_t>(*tensor);
        for (uint8_t value : result) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    } else if (tensor->type == kTfLiteFloat32) {
        std::cout << "output is float32 " << std::endl;
        absl::Span<const float> result = coral::TensorData<float>(*tensor);
        for (float value : result) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        // const float* output_data = coral::MutableTensorData<float>(*tensor);
        // int output_size = coral::GetTensorSize(*tensor);
        // for (int i = 0; i < output_size; ++i) {
        //     std::cout << output_data[i] << " ";
        // }
        std::cout << std::endl;
    } else {
        std::cerr << "Unsupported output tensor type" << std::endl;
    }
    

    return 0;
}
