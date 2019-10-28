#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matmul_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatmulLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    M_ = bottom[0]->num();          //rows of A
    N_ = bottom[1]->num();          //cols of B
    K_ = bottom[0]->channels();     //rows of A = cols of B
    W_ = sqrt(K_);
}

template <typename Dtype>
void MatmulLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int myints[] = {M_, M_};
  vector<int> top_shape (myints, myints + sizeof(myints) / sizeof(int) );
  top[0]->Reshape(top_shape); //reshape top
}

template <typename Dtype>
void MatmulLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = bottom[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                            M_, N_, K_, (Dtype)1.,
                            bottom_data, weight,
                            (Dtype)0., top_data);
  }
}

template <typename Dtype>
void MatmulLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* weight = bottom[0]->cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            M_, K_, N_, (Dtype)1.,
                            top_diff, weight，
                            (Dtype)0., bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(MatmulLayer);
#endif

INSTANTIATE_CLASS(MatmulLayer);
REGISTER_LAYER_CLASS(MatmulLayer);

}  // namespace caffe
