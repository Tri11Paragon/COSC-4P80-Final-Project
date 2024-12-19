/*
 *  <Short Description>
 *  Copyright (C) 2024  Brett Terpstra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include <MNIST.h>
#include <blt/fs/loader.h>
#include <blt/std/memory.h>
#include <blt/std/memory_util.h>
#include <variant>
#include <blt/iterator/iterator.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>

namespace fp
{
    class idx_file_t
    {
        template <typename T>
        using mk_v = std::vector<T>;
        using vec_t = std::variant<mk_v<blt::u8>, mk_v<blt::i8>, mk_v<blt::u16>, mk_v<blt::u32>, mk_v<blt::f32>, mk_v<blt::f64>>;

    public:
        explicit idx_file_t(const std::string& path)
        {
            std::ifstream file{path, std::ios::in | std::ios::binary};

            using char_type = std::ifstream::char_type;
            char_type magic_arr[4];
            file.read(magic_arr, 4);
            BLT_ASSERT(magic_arr[0] == 0 && magic_arr[1] == 0);

            blt::u8 dims = magic_arr[3];
            blt::size_t total_size = 1;

            for (blt::i32 i = 0; i < dims; i++)
            {
                char_type dim_arr[4];
                file.read(dim_arr, 4);
                blt::u32 dim;
                blt::mem::fromBytes(dim_arr, dim);
                dimensions.push_back(dim);
                total_size *= dim;
            }

            switch (magic_arr[2])
            {
            // unsigned char
            case 0x08:
                data = mk_v<blt::u8>{};
                read_data<blt::u8>(file, total_size);
                break;
            // signed char
            case 0x09:
                data = mk_v<blt::i8>{};
                read_data<blt::i8>(file, total_size);
                break;
            // short
            case 0x0B:
                data = mk_v<blt::u16>{};
                read_data<blt::u16>(file, total_size);
                reverse_data<blt::u16>();
                break;
            // int
            case 0x0C:
                data = mk_v<blt::u32>{};
                read_data<blt::u32>(file, total_size);
                reverse_data<blt::u32>();
                break;
            // float
            case 0x0D:
                data = mk_v<blt::f32>{};
                read_data<blt::f32>(file, total_size);
                reverse_data<blt::f32>();
                break;
            // double
            case 0x0E:
                data = mk_v<blt::f64>{};
                read_data<blt::f64>(file, total_size);
                reverse_data<blt::f64>();
                break;
            default:
                BLT_ERROR("Unspported idx file type!");
            }
            if (file.eof())
            {
                BLT_ERROR("EOF reached. It's unlikely your file was read correctly!");
            }
        }

        template <typename T>
        [[nodiscard]] const std::vector<T>& get_data_as() const
        {
            return std::get<mk_v<T>>(data);
        }

        template <typename T>
        std::vector<blt::span<T>> get_as_spans() const
        {
            std::vector<blt::span<T>> spans;

            blt::size_t total_size = data_size(1);

            for (blt::size_t i = 0; i < dimensions[0]; i++)
            {
                auto& array = std::get<mk_v<T>>(data);
                spans.push_back({&array[i * total_size], total_size});
            }

            return spans;
        }

        [[nodiscard]] const std::vector<blt::u32>& get_dimensions() const
        {
            return dimensions;
        }

        [[nodiscard]] blt::size_t data_size(const blt::size_t starting_dimension = 0) const
        {
            blt::size_t total_size = 1;
            for (const auto d : blt::iterate(dimensions).skip(starting_dimension))
                total_size *= d;
            return total_size;
        }

    private:
        template <typename T>
        void read_data(std::ifstream& file, blt::size_t total_size)
        {
            auto& array = std::get<mk_v<T>>(data);
            array.resize(total_size);
            file.read(reinterpret_cast<char*>(array.data()), static_cast<std::streamsize>(total_size) * sizeof(T));
        }

        template <typename T>
        void reverse_data()
        {
            auto& array = std::get<mk_v<T>>(data);
            for (auto& v : array)
                blt::mem::reverse(v);
        }

        std::vector<blt::u32> dimensions;
        vec_t data;
    };

    class image_t
    {
    public:
        static constexpr blt::u32 target_size = 10;

        image_t(const idx_file_t& image_data, const idx_file_t& label_data): samples(image_data.get_dimensions()[0]),
                                                                             input_size(image_data.data_size(1))
        {
            BLT_ASSERT_MSG(samples == label_data.get_dimensions()[0],
                           ("Mismatch in data sample sizes! " + std::to_string(samples) + " vs " + std::to_string(label_data.get_dimensions()[0])).
                           c_str());
            auto& image_array = image_data.get_data_as<blt::u8>();
            auto& label_array = label_data.get_data_as<blt::u8>();

            for (const auto label : label_array)
                image_labels.push_back(label);

            const auto row_length = image_data.get_dimensions()[2];
            const auto number_of_rows = image_data.get_dimensions()[1];

            for (blt::u32 i = 0; i < samples; i++)
            {
                dlib::matrix<blt::u8> mat(number_of_rows, row_length);
                for (blt::u32 y = 0; y < number_of_rows; y++)
                {
                    for (blt::u32 x = 0; x < row_length; x++)
                    {
                        mat(x, y) = image_array[i * input_size + y * row_length + x];
                    }
                }
                data.push_back(mat);
            }
        }

        [[nodiscard]] const std::vector<dlib::matrix<blt::u8>>& get_image_data() const
        {
            return data;
        }

        [[nodiscard]] const std::vector<blt::u64>& get_image_labels() const
        {
            return image_labels;
        }

    private:
        blt::u32 samples;
        blt::u32 input_size;
        std::vector<dlib::matrix<blt::u8>> data;
        std::vector<blt::u64> image_labels;
    };

    void run_mnist()
    {
        using namespace dlib;

        idx_file_t test_images{"../problems/mnist/t10k-images.idx3-ubyte"};
        idx_file_t test_labels{"../problems/mnist/t10k-labels.idx1-ubyte"};
        idx_file_t train_images{"../problems/mnist/train-images.idx3-ubyte"};
        idx_file_t train_labels{"../problems/mnist/train-labels.idx1-ubyte"};

        auto train_samples = train_images.get_dimensions()[0];
        auto test_samples = test_images.get_dimensions()[0];

        auto columns = train_images.get_dimensions()[1];
        auto rows = train_images.get_dimensions()[2];

        auto input_size = static_cast<blt::u32>(train_images.data_size(1));

        image_t train_image{train_images, train_labels};
        image_t test_image{test_images, test_labels};

        // using net_type = loss_multiclass_log<
        //                                 fc<10,
        //                                 relu<fc<84,
        //                                 relu<fc<120,
        //                                 max_pool<2,2,2,2,relu<con<16,5,5,1,1,
        //                                 max_pool<2,2,2,2,relu<con<6,5,5,1,1,
        //                                 input<matrix<blt::u8>>>>>>>>>>>>>>;

        using net_type = loss_multiclass_log<
                                        fc<10,
                                        sig<fc<84,
                                        sig<fc<120,
                                        input<matrix<blt::u8>>>>>>>>;

        net_type network{};

        dnn_trainer trainer(network);
        trainer.set_learning_rate(0.01);
        trainer.set_min_learning_rate(0.00001);
        trainer.set_mini_batch_size(128);
        trainer.be_verbose();

        // trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));

        trainer.train(train_image.get_image_data(), train_image.get_image_labels());

        network.clean();
        serialize("mnist_network.dat") << network;

        std::vector<unsigned long> predicted_labels = network(train_image.get_image_data());
        int num_right = 0;
        int num_wrong = 0;
        // And then let's see if it classified them correctly.
        for (size_t i = 0; i < train_image.get_image_data().size(); ++i)
        {
            if (predicted_labels[i] == train_image.get_image_labels()[i])
                ++num_right;
            else
                ++num_wrong;

        }
        std::cout << "training num_right: " << num_right << std::endl;
        std::cout << "training num_wrong: " << num_wrong << std::endl;
        std::cout << "training accuracy:  " << num_right / static_cast<double>(num_right + num_wrong) << std::endl;

        predicted_labels = network(test_image.get_image_data());
        num_right = 0;
        num_wrong = 0;
        for (size_t i = 0; i < test_image.get_image_data().size(); ++i)
        {
            if (predicted_labels[i] == test_image.get_image_labels()[i])
                ++num_right;
            else
                ++num_wrong;

        }
        std::cout << "testing num_right: " << num_right << std::endl;
        std::cout << "testing num_wrong: " << num_wrong << std::endl;
        std::cout << "testing accuracy:  " << num_right / static_cast<double>(num_right + num_wrong) << std::endl;
    }
}
