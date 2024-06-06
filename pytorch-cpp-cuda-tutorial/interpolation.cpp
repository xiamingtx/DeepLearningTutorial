/*
 * @Description: Description of this file
 * @Version: 2.0
 * @Author: 夏明
 * @Date: 2024-06-06 14:57:45
 * @LastEditors: 夏明
 * @LastEditTime: 2024-06-06 15:27:49
 */
#include <torch/extension.h>

torch::Tensor trilinear_interpolation(torch::Tensor feats, torch::Tensor point)
{
    return feats;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("trilinear_interpolation", &trilinear_interpolation);
}
