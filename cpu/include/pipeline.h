#ifndef CNN_PIPELINE_H
#define CNN_PIPELINE_H



// C++
#include <map>
#include <random>
#include <filesystem>
// self
#include "data_format.h"


namespace pipeline {

    using list_type = std::vector<std::pair<std::string, int> >;
    // From the folder dataset_path, get a list of images of different categories according to categories
    std::map<std::string, list_type> get_images_for_classification(
            const std::filesystem::path dataset_path,
            const std::vector<std::string> categories={},
            const std::pair<float, float> ratios={0.8, 0.1});

    // This is very slow!!!!, if there is data enhancement, the speed will be reduced to 1/4 of the original
    class ImageAugmentor {
    private:
        std::default_random_engine e, l, c, r; // e is used to get the probability of the operation; l is used to shuffle the operation list; c is used to get the probability required for clipping; r is used to get the probability of rotation
        std::uniform_real_distribution<float> engine;
        std::uniform_real_distribution<float> crop_engine;
        std::uniform_real_distribution<float> rotate_engine;
        std::uniform_int_distribution<int> minus_engine;
        std::vector<std::pair<std::string, float> > ops;
    public:
        ImageAugmentor(const std::vector<std::pair<std::string, float> >& _ops={{"hflip", 0.5}, {"vflip", 0.2}, {"crop", 0.7}, {"rotate", 0.5}})
        : e(212), l(826), c(320), r(520),
        engine(0.0, 1.0), crop_engine(0.0, 0.25), rotate_engine(15, 75), minus_engine(1, 10),
        ops(std::move(_ops)) {}
        void make_augment(cv::Mat& origin, const bool show=false);
    };

    class DataLoader {
    using batch_type = std::pair< std::vector<tensor>, std::vector<int> >; // batch is a pair
    private:
        list_type images_list; // Dataset list, image <==> label
        int images_num; // How many images and corresponding labels are there in this sub-dataset
        const int batch_size; // Pack several images at a time
        const bool augment; // Do you want to do image enhancement
        const bool shuffle; // Do you want to shuffle the list
        const int seed; // Randomly shuffle the seed of the list each time
        int iter = -1; // The iter image is currently collected
        std::vector<tensor> buffer; // batch buffer, used to generate tensor from image, does not need to be allocated and destroyed every time the image is to be read
        const int H, W, C; // Allowed image size
    public:
        explicit DataLoader(const list_type& _images_list, const int _bs=1, const bool _aug=false, const bool _shuffle=true, const std::tuple<int, int, int> image_size={224, 224, 3}, const int _seed=212);
        int length() const;
        batch_type generate_batch();
        private:
        // Get the image information of the batch_index and fill it into tensor
        std::pair<tensor, int> add_to_buffer(const int batch_index);
        // Image enhancement
        ImageAugmentor augmentor;
    };
}


#endif //CNN_PIPELINE_H
