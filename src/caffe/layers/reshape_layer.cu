#include <vector>

#include "caffe/layers/reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReshapeForward(const int nthreads, const Dtype* bottom_data,
    Dtype* top_data, const int bc, const int bh, const int bw, const int tc,
    const int th, const int tw, const int ps) {
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int old_n = idx/bc/bh/bw;
    const int old_c = (idx/bh/bw)%bc;
    const int old_h = (idx/bw)%bh;
    const int old_w = idx%bw;
    const int new_c = old_c/(ps*ps);
    const int new_h = old_h*ps + (old_c%(ps*ps))/ps;
    const int new_w = old_w*ps + (old_c%(ps*ps))%ps;
    int top_index = ((old_n*tc + new_c)*th + new_h)*tw + new_w;
    top_data[top_index] = bottom_data[idx];
  }
}

template <typename Dtype>
__global__ void ReshapeBackward(const int nthreads, Dtype* bottom_diff,
    const Dtype* top_diff, const int bc, const int bh, const int bw,
    const int tc, const int th, const int tw, const int ps) {
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int new_n = idx/tc/th/tw;
    const int new_c = (idx/th/tw)%tc;
    const int new_h = (idx/tw)%th;
    const int new_w = idx%tw;
    const int old_c = new_c*ps*ps + (new_h%ps)*ps + new_w%ps;
    const int old_h = new_h/ps;
    const int old_w = new_w/ps;
    const int bottom_index = ((new_n*bc + old_c)*bh + old_h)*bw + old_w;
    bottom_diff[bottom_index] = top_diff[idx];
  }
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int ps = this->layer_param_.reshape_param().pixelshuffler();
  if (ps != 0) {
    const int count = bottom[0]->count();
    vector<int> bottom_shape = bottom[0]->shape();
    const int bc = bottom_shape[1];
    const int bh = bottom_shape[2];
    const int bw = bottom_shape[3];
    vector<int> top_shape = top[0]->shape();
    const int tc = top_shape[1];
    const int th = top_shape[2];
    const int tw = top_shape[3];
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    ReshapeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data, bc, bh, bw, tc, th, tw, ps);
  }
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int ps = this->layer_param_.reshape_param().pixelshuffler();
  if (ps != 0) {
    const int count = top[0]->count();
    vector<int> top_shape = top[0]->shape();
    const int tc = top_shape[1];
    const int th = top_shape[2];
    const int tw = top_shape[3];
    vector<int> bottom_shape = bottom[0]->shape();
    const int bc = bottom_shape[1];
    const int bh = bottom_shape[2];
    const int bw = bottom_shape[3];
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->gpu_diff();
    ReshapeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_diff, top_diff, bc, bh, bw, tc, th, tw, ps);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReshapeLayer);

}  // namespace caffe
