#include <cmath>
#include <vector>

#include "caffe/layers/swish_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype Xsigmoid(Dtype x) {
  return x / (1. + exp(-x));
}

template <typename Dtype>
void SwishLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  if (bottom[0] == top[0]) {
    // For in-place computation
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SwishLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }
  for (int i = 0; i < count; ++i) {
    top_data[i] = Xsigmoid(bottom_data[i]);
  }
}

template <typename Dtype>
void SwishLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // For in-place computation
    if (top[0] == bottom[0]) {
      bottom_data = bottom_memory_.cpu_data();
    }
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype Xsigmoid_x = top_data[i];
      const Dtype sigmoid_x = sigmoid(bottom_data[i]);
      bottom_diff[i] = top_diff[i] * ( (Xsigmoid_x + sigmoid_x * (1. - Xsigmoid_x)) );
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SwishLayer);
#endif

INSTANTIATE_CLASS(SwishLayer);
REGISTER_LAYER_CLASS(Swish);

}  // namespace caffe
