// Copyright 2022 Sony Group Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nbla/singleton_manager.hpp>
#include <nbla_utils/nnp.hpp>

#ifdef WITH_CUDA
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/cuda/init.hpp>
#endif

#ifdef TIMING
#include <chrono>
#endif

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>

#include <curl/curl.h>
#include <stdio.h>

#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std;

#define URL                                                                    \
  "https://nnabla.org/pretrained-models/nnp_models/semantic_segmentation/"
#define DEFAULT_PATH "nnabla_data/nnp_models/semantic_segmentation/"

size_t write_data(void *ptr, size_t size, size_t n_mem, FILE *f_stream) {
  size_t n_write = fwrite(ptr, size, n_mem, f_stream);
  return n_write;
}

void split_path_file(const string &input, string &path, string &file) {
  path.empty();
  file.empty();
  size_t pos = input.rfind("/");
  if (pos == string::npos) {
    file = input;
  } else {
    file = input.substr(pos + 1, input.length() - pos - 1);
    path = input.substr(0, pos + 1);
  }
  return;
}

void curl_download_nnp(const string &nnp_pos) {
  string path, nnp_name;
  split_path_file(nnp_pos, path, nnp_name);
  string s_url = URL + nnp_name;
  cout << "Download " << nnp_name << " from:" << endl
       << s_url << endl
       << "The file will be saved at:" << endl
       << nnp_pos << endl
       << "Downloading..." << endl;
  CURL *p_curl;
  FILE *p_f;
  CURLcode ret;
  p_curl = curl_easy_init();
  if (p_curl) {
    p_f = fopen(nnp_pos.c_str(), "wb");
    curl_easy_setopt(p_curl, CURLOPT_URL, s_url.c_str());
    curl_easy_setopt(p_curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(p_curl, CURLOPT_WRITEDATA, p_f);
    ret = curl_easy_perform(p_curl);
    cout << "curl return code is:" << ret << endl;
    curl_easy_cleanup(p_curl);
    fclose(p_f);
  }
}

void check_arg_nnp_file(const string &arg_nnp_file, const string &default_name,
                        string &output_file, string &nnp_name) {
  vector<string> vs_all_nnp_name{
      "DeepLabV3-voc-os-16.nnp", "DeepLabV3-voc-os-8.nnp",
      "DeepLabV3-voc-coco-os-16.nnp", "DeepLabV3-voc-coco-os-8.nnp"};
  string nnp_path;
  split_path_file(arg_nnp_file, nnp_path, nnp_name);
  if (vs_all_nnp_name.end() ==
      find(vs_all_nnp_name.begin(), vs_all_nnp_name.end(), nnp_name)) {
    cout << "nnp file name must one of bellow:" << endl;
    for (auto &e : vs_all_nnp_name) {
      cout << e << endl;
    }
    cout << endl;
    cout << "Set nnp file name to default: " << default_name << endl << endl;
    nnp_name = default_name;
  }
  output_file = nnp_path + nnp_name;
  return;
}

bool file_exist(const string &file) {
  const ifstream f(file.c_str());
  return f.good();
}

int main(int argc, char *argv[]) {
  if (!(argc >= 3)) {
    cerr << "Usage: " << argv[0] << " nnp_file input_picture" << endl;
    cerr << "The default argument: " << argv[0]
         << " ./DeepLabV3-voc-coco-os-16.nnp.nnp"
         << " ./test.jpg" << endl;
    cerr << endl;
    cerr << "Positional arguments: " << endl;
    cerr << "  nnp_file  : .nnp file" << endl;
    cerr << "  input_picture :  picture to sematic" << endl;
    cerr << endl;
  }
  const string s_default_name = "DeepLabV3-voc-coco-os-16.nnp";
  string arg_nnp_file("./DeepLabV3-voc-coco-os-16.nnp");
  string arg_picture("./test.jpg");
  if (argc >= 2) {
    arg_nnp_file = argv[1];
  }
  if (argc >= 3) {
    arg_picture = argv[2];
  }
  string input_nnp_file;
  string nnp_name;
  check_arg_nnp_file(arg_nnp_file, s_default_name, input_nnp_file, nnp_name);

  if (!file_exist(input_nnp_file)) {
    cout << "Not find existing nnp file from given path ..." << endl
         << "Find nnp file from default path." << endl;
    struct passwd *pw = getpwuid(getuid());
    const string home_dir = pw->pw_dir;
    string pos_to_save = home_dir + "/" + DEFAULT_PATH + nnp_name;
    if (!file_exist(pos_to_save)) {
      cout << "Not find existing nnp file from default path ..." << endl;
      curl_download_nnp(pos_to_save);
    } else {
      input_nnp_file = pos_to_save;
    }
  }
  if (!file_exist(arg_picture)) {
    cout << "Please specify a picture to semantic" << endl;
    return -1;
  }

  string executor_name("runtime");
  if (argc == 4) {
    executor_name = argv[3];
  }

  cout << endl << "Execute infer ..." << endl;
  // Create a context (the following setting is recommended.)
  nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
#ifdef WITH_CUDA
  cout << "With CUDA" << endl;
  nbla::init_cudnn();
  nbla::Context ctx{
      {"cudnn:float", "cuda:float", "cpu:float"}, "CudaCachedArray", "0"};
#else
  nbla::Context ctx = cpu_ctx;
#endif

  // Create a Nnp object
  nbla::utils::nnp::Nnp nnp(ctx);

  // Set nnp file to Nnp object.
  nnp.add(input_nnp_file);

  // Get an executor instance.
  auto executor = nnp.get_executor(executor_name);
  executor->set_batch_size(1); // Use batch_size = 1.

  // Get input data as a CPU array.
  nbla::CgVariablePtr x = executor->get_data_variables().at(0).variable;
  float *data = x->variable()->cast_data_and_get_pointer<float>(cpu_ctx);
  // print x variable info
  {
    printf("x variable shape:");
    for (auto e : x->variable()->shape()) {
      cout << e << " ";
    }
    printf(" size:%ld", x->variable()->size());
    cout << " type: " << typeid(*data).name() << endl;
  }

  // Read image by opencv
  cv::Mat in_image, image_src, image_tmp;
  image_src = cv::imread(arg_picture, cv::IMREAD_COLOR);
  int H = 513, W = 513;
  int size_x = x->variable()->shape().size();
  if (size_x >= 2) {
    W = x->variable()->shape().at(size_x - 1);
    H = x->variable()->shape().at(size_x - 2);
  }
  cv::Size size = cv::Size(H, W);
  cv::resize(image_src, in_image, size, 0, 0, cv::INTER_LINEAR);
  // show the input image
  {
    cv::namedWindow("origin_image", cv::WINDOW_AUTOSIZE);
    cv::imshow("origin_image", image_src);
    cv::namedWindow("image to input model", cv::WINDOW_AUTOSIZE);
    cv::imshow("image to input model", in_image);
  }
  // BGR2RGB, the format in python is RGB
  cv::cvtColor(in_image, in_image, cv::COLOR_BGR2RGB);

  // transpos data and unify input data to [-1,1]
  vector<float> vpf(3 * H * W);
  float *pf = vpf.data();
  {
    vector<u_char> vd0(H * W);
    vector<u_char> vd1(H * W);
    vector<u_char> vd2(H * W);
    u_char *d0 = vd0.data();
    u_char *d1 = vd1.data();
    u_char *d2 = vd2.data();
    for (int i = 0; i < H * W; ++i) {
      d0[i] = (u_char)(*(in_image.data + i * 3 + 0));
      d1[i] = (u_char)(*(in_image.data + i * 3 + 1));
      d2[i] = (u_char)(*(in_image.data + i * 3 + 2));
    }
    for (int i = 0; i < H * W; ++i) {
      *(pf + H * W * 0 + i) = (2.0 / 255.0) * d0[i] - 1.0;
      *(pf + H * W * 1 + i) = (2.0 / 255.0) * d1[i] - 1.0;
      *(pf + H * W * 2 + i) = (2.0 / 255.0) * d2[i] - 1.0;
    }
  }

  // input date to x variable
  memcpy(data, pf, 1 * 3 * H * W * sizeof(*pf));

// executor->execute();
#ifdef WITH_CUDA
  nbla::cuda_device_synchronize("0");
#endif
  // Timing starts
  auto start = chrono::steady_clock::now();

  // Execute prediction
  cout << "Executing..." << endl;
  executor->execute();

#ifdef WITH_CUDA
  nbla::cuda_device_synchronize("0");
#endif
  // Timing ends
  auto end = chrono::steady_clock::now();
  cout << "Elapsed time: "
       << chrono::duration_cast<chrono::microseconds>(end - start).count() *
              0.001
       << " [ms]." << endl;

  // Get output as a CPU array;
  nbla::CgVariablePtr y = executor->get_output_variables().at(0).variable;
  const float *y_data = y->variable()->get_data_pointer<float>(cpu_ctx);
  // print y_data info
  {
    printf("y variable shape:");
    for (auto e : y->variable()->shape()) {
      cout << e << " ";
    }
    printf("size:%ld", y->variable()->size());
    cout << " type: " << typeid(*y_data).name() << endl;
  }

  // np.argmax(y,axis=1)
  u_char uch_ar_result[H * W];
  {
    for (int i = 0; i < H * W; ++i) {
      float max = 0.0;
      int index = 0;
      for (int j = 0; j < 21; ++j) {
        if (*(y_data + i + H * W * j) > max) {
          max = *(y_data + i + H * W * j);
          index = j;
        }
      }
      uch_ar_result[i] = index;
    }
  }

  // make color and show the segmented picture
  u_char uch_ar_color[][3] = {
      {0, 0, 0},    {128, 0, 0},     {0, 128, 0},    {128, 128, 0},
      {0, 0, 128},  {120, 0, 128},   {0, 128, 128},  {128, 128, 128},
      {64, 0, 0},   {192, 0, 0},     {64, 128, 0},   {192, 128, 0},
      {64, 0, 128}, {192, 0, 128},   {64, 128, 128}, {192, 128, 128},
      {0, 64, 0},   {128, 64, 0},    {0, 192, 0},    {128, 192, 0},
      {0, 64, 128}, {224, 224, 192}, {0, 0, 0}};
  {
    image_tmp = in_image;
    for (int i = 0; i < H * W; ++i) {
      image_tmp.data[i * 3 + 0] = uch_ar_color[uch_ar_result[i]][0];
      image_tmp.data[i * 3 + 1] = uch_ar_color[uch_ar_result[i]][1];
      image_tmp.data[i * 3 + 2] = uch_ar_color[uch_ar_result[i]][2];
    }

    cv::Mat image_resize(image_src.size(), CV_8UC3);
    cv::resize(image_tmp, image_resize, image_resize.size(), 0, 0,
               cv::INTER_LINEAR);
    cv::namedWindow("segmented image");
    cv::imshow("segmented image", image_resize);
  }

  // show the result of segmented:
  const string category[] = {
      "aeroplane", "bicycle",   "bird",   "boat",        "bottle",      "bus",
      "car",       "cat",       "chair",  "cow",         "diningtable", "dog",
      "horse",     "motorbike", "person", "pottedplant", "sheep",       "sofa",
      "train",     "tvmonitor", "unknow"};
  vector<int> vi_result;
  {
    for (auto e : uch_ar_result) {
      if (find(vi_result.begin(), vi_result.end(), e) == vi_result.end()) {
        vi_result.push_back(e);
      }
    }
    printf("show the result of segmentation:\n");
    for (auto e : vi_result) {
      cout << "Classes Segmented: " << category[e] << endl;
    }
  }

  cout << "Press any key on the image to exit ..." << endl;
  cv::waitKey();

  nbla::SingletonManager::clear();

  return 0;
}
