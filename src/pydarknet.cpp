#include <iostream>
#include <boost/python.hpp>
#include <Python.h>
#include <vector>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "option_list.h"
#include "region_layer.h"

#ifdef GPU
#include "cuda.h"
#endif

using namespace std;

typedef struct BBox{
    int left;
    int right;
    int top;
    int bottom;
    float confidence;
    int cls;
} _BBox;

int max_i(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

void draw_detections_bbox(image im, int num, float thresh, box *boxes, float **probs, int classes, vector<_BBox> &bb)
{
    int i;

    for(i = 0; i < num; ++i){
        int cla = max_i(probs[i], classes);
        float prob = probs[i][cla];
        if(prob > thresh){

            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            _BBox bs;
            bs.left = left; bs.right = right; bs.top = top; bs.bottom = bot; bs.confidence = prob; bs.cls = cla;
            bb.push_back(bs);
        }
    }
}

class DarknetObjectDetector{
public:
    DarknetObjectDetector(boost::python::str cfg_name, boost::python::str weight_name){

        string cfg_c_name = string(((const char*)boost::python::extract<const char*>(cfg_name)));
        string weight_c_name = string(((const char*)boost::python::extract<const char*>(weight_name)));

        cout << "loading network spec from" << cfg_c_name << '\n';
        net = parse_network_cfg((char*)cfg_c_name.c_str());

        cout << "loading network weights from" << weight_c_name << '\n';
        load_weights(&net, (char*)weight_c_name.c_str());

        cout << "network initialized!\n";
        set_batch_network(&net, 1); srand(2222222);

        thresh = 0.2;
    };

    ~DarknetObjectDetector()
    {
    };

    boost::python::list detect_object(boost::python::str img_data, int img_width, int img_height, int img_channel){

        // preprocess input image
        const unsigned char* data = (const unsigned char*)((const char*)boost::python::extract<const char*>(img_data));
        boost::python::list ret_list = boost::python::list();
        vector<_BBox> bboxes;

        assert(img_channel == 3);
        image im = make_image(img_width, img_height, img_channel);

        int cnt = img_height * img_channel * img_width;
        for (int i = 0; i < cnt; ++i){
            im.data[i] = (float)data[i] / 255.;
        }

        image sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n-1];

        box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
        for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

        float *X = sized.data;
        network_predict(net, X);
        float nms = .4f;

        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
        if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections_bbox(im, l.w*l.h*l.n, thresh, boxes, probs, l.classes, bboxes);

        save_image(im, "temp");

        free_image(im);
        free_image(sized);

        for (int i = 0; i < bboxes.size(); i++)
        {
            ret_list.append<BBox>(bboxes[i]);
        }


        return ret_list;
    };

    static void set_device(int dev_id){
#ifdef GPU
        cudaError_t err = cudaSetDevice(dev_id);
        if (err != cudaSuccess){
            cout << "CUDA Error on setting device: " << cudaGetErrorString(err) << '\n';
            PyErr_SetString(PyExc_Exception, "Not able to set device");
        }
#else
        PyErr_SetString(PyExc_Exception, "Not compiled with CUDA");
#endif
    }

private:

    network net;
//	detection_layer layer;
    char **names;
    float thresh;
};

BOOST_PYTHON_MODULE(libpydarknet)
{
    using namespace boost::python;
    class_<DarknetObjectDetector>("DarknetObjectDetector", init<str, str>())
        .def("detect_object", &DarknetObjectDetector::detect_object)
        .def("set_device", &DarknetObjectDetector::set_device)
        .staticmethod("set_device");

    class_<BBox>("BBox")
        .def_readonly("left", &BBox::left)
        .def_readonly("right", &BBox::right)
        .def_readonly("top", &BBox::top)
        .def_readonly("bottom", &BBox::bottom)
        .def_readonly("confidence", &BBox::confidence)
        .def_readonly("cls", &BBox::cls);
}

//char const *greet() {
//  return "hello world";
//}

//#include <boost/python.hpp>

//BOOST_PYTHON_MODULE(libpydarknet) {
//  using namespace boost::python;
//  def("greet", greet);
//}
