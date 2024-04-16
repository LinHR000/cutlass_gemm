#include <torch/extension.h>
#include <iostream>
#include <cassert>
#include <string.h>
#include "tensorrt_llm/kernels/fp8_dtype_converter/fp8_quanttools.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_quant_fp8", &fake_quant_fp8, "fake_quant_fp8_e5m2");
  m.def("fp8_dtype_converter", &convert_fp8_e5m2, "convert_fp8_e5m2");
}

