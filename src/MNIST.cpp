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
#include <filesystem>
#include <iomanip>
#include <blt/iterator/iterator.h>
#include <blt/parse/argparse.h>
#include <blt/std/time.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <csignal>

namespace fp
{
    constexpr blt::i64 batch_size = 256;

    std::string binary_directory;
    std::string python_dual_stacked_graph_program;
    std::atomic_bool break_flag = false;
    std::atomic_bool stop_flag = false;
    std::atomic_bool learn_flag = false;
    std::atomic_int64_t last_epoch = -1;

    void run_python_line_graph(const std::string& title, const std::string& output_file, const std::string& csv1, const std::string& csv2,
                               const blt::size_t pos_forward, const blt::size_t pos_deep)
    {
        const auto command = "python3 '" + python_dual_stacked_graph_program + "' '" + title + "' '" + output_file + "' '" + csv1 + "' '" + csv2 + "' "
            + std::to_string(pos_forward) + " " + std::to_string(pos_deep);
        BLT_TRACE("Running %s", command.c_str());
        std::system(command.c_str());
    }

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
        using data_iterator = std::vector<dlib::matrix<blt::u8>>::const_iterator;
        using label_iterator = std::vector<blt::u64>::const_iterator;

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

    struct batch_stats_t
    {
        blt::u64 hits = 0;
        blt::u64 misses = 0;

        friend std::ofstream& operator<<(std::ofstream& file, const batch_stats_t& stats)
        {
            file << stats.hits << ',' << stats.misses;
            return file;
        }

        friend std::ifstream& operator>>(std::ifstream& file, batch_stats_t& stats)
        {
            file >> stats.hits;
            file.ignore();
            file >> stats.misses;
            return file;
        }

        batch_stats_t& operator+=(const batch_stats_t& stats)
        {
            hits += stats.hits;
            misses += stats.misses;
            return *this;
        }

        batch_stats_t& operator/=(const blt::u64 divisor)
        {
            hits /= divisor;
            misses /= divisor;
            return *this;
        }
    };

    struct epoch_stats_t
    {
        batch_stats_t test_results{};
        double average_loss = 0;
        double learn_rate = 0;

        friend std::ofstream& operator<<(std::ofstream& file, const epoch_stats_t& stats)
        {
            file << stats.test_results << ',' << stats.average_loss << ',' << stats.learn_rate;
            return file;
        }

        friend std::ifstream& operator>>(std::ifstream& file, epoch_stats_t& stats)
        {
            file >> stats.test_results;
            file.ignore();
            file >> stats.average_loss;
            file.ignore();
            file >> stats.learn_rate;
            return file;
        }

        epoch_stats_t& operator+=(const epoch_stats_t& stats)
        {
            test_results += stats.test_results;
            average_loss += stats.average_loss;
            learn_rate += stats.learn_rate;
            return *this;
        }

        epoch_stats_t& operator/=(const blt::u64 divisor)
        {
            test_results /= divisor;
            average_loss /= static_cast<double>(divisor);
            learn_rate /= static_cast<double>(divisor);
            return *this;
        }
    };

    struct network_stats_t
    {
        std::vector<epoch_stats_t> epoch_stats;

        friend std::ofstream& operator<<(std::ofstream& file, const network_stats_t& stats)
        {
            file << stats.epoch_stats.size();
            file << '\n';
            for (const auto& v : stats.epoch_stats)
                file << v << "\n";
            return file;
        }

        friend std::ifstream& operator>>(std::ifstream& file, network_stats_t& stats)
        {
            blt::size_t size;
            file >> size;
            file.ignore();
            for (blt::size_t i = 0; i < size; i++)
            {
                stats.epoch_stats.emplace_back();
                file >> stats.epoch_stats.back();
                file.ignore();
            }
            return file;
        }
    };

    struct network_average_stats_t
    {
        std::vector<network_stats_t> run_stats;

        network_average_stats_t& operator+=(const network_stats_t& stats)
        {
            run_stats.push_back(stats);
            return *this;
        }

        [[nodiscard]] blt::size_t average_size() const
        {
            blt::size_t acc = 0;
            for (const auto& [epoch_stats] : run_stats)
                acc += epoch_stats.size();
            return acc;
        }

        [[nodiscard]] network_stats_t average_stats() const
        {
            network_stats_t stats;
            for (const auto& [epoch_stats] : run_stats)
            {
                if (stats.epoch_stats.size() < epoch_stats.size())
                    stats.epoch_stats.resize(epoch_stats.size());
                for (const auto& [i, v] : blt::enumerate(epoch_stats))
                {
                    stats.epoch_stats[i] += v;
                }
            }
            for (auto& v : stats.epoch_stats)
                v /= run_stats.size();
            return stats;
        }

        friend std::ofstream& operator<<(std::ofstream& file, const network_average_stats_t& stats)
        {
            file << stats.run_stats.size();
            file << '\n';
            for (const auto& v : stats.run_stats)
                file << v << "---\n";
            return file;
        }

        friend std::ifstream& operator>>(std::ifstream& file, network_average_stats_t& stats)
        {
            blt::size_t size;
            file >> size;
            file.ignore();
            for (blt::size_t i = 0; i < size; i++)
            {
                stats.run_stats.emplace_back();
                file >> stats.run_stats.back();
                file.ignore(4);
            }
            return file;
        }
    };

    template <blt::i64 batch_size = batch_size, typename NetworkType>
    batch_stats_t test_batch(NetworkType& network, image_t::data_iterator begin, const image_t::data_iterator end, image_t::label_iterator lbegin)
    {
        batch_stats_t stats{};

        std::array<image_t::label_iterator::value_type, batch_size> output_labels{};

        auto amount_remaining = std::distance(begin, end);

        while (amount_remaining != 0)
        {
            const auto batch = std::min(amount_remaining, batch_size);
            network(begin, begin + batch, output_labels.begin());

            for (auto [predicted, expected] : blt::iterate(output_labels.begin(), output_labels.begin() + batch).zip(lbegin, lbegin + batch))
            {
                if (predicted == expected)
                    ++stats.hits;
                else
                    ++stats.misses;
            }

            begin += batch;
            lbegin += batch;
            amount_remaining -= batch;
        }

        return stats;
    }

    template <typename NetworkType>
    batch_stats_t test_network(NetworkType& network)
    {
        const idx_file_t test_images{binary_directory + "../problems/mnist/t10k-images.idx3-ubyte"};
        const idx_file_t test_labels{binary_directory + "../problems/mnist/t10k-labels.idx1-ubyte"};

        const image_t test_image{test_images, test_labels};

        auto test_results = test_batch(network, test_image.get_image_data().begin(), test_image.get_image_data().end(),
                                       test_image.get_image_labels().begin());

        BLT_DEBUG("Testing hits: %lu", test_results.hits);
        BLT_DEBUG("Testing misses: %lu", test_results.misses);
        BLT_DEBUG("Testing accuracy: %lf", test_results.hits / static_cast<double>(test_results.hits + test_results.misses));

        return test_results;
    }

    template <typename NetworkType>
    network_stats_t train_network(const std::string& ident, NetworkType& network)
    {
        const idx_file_t train_images{binary_directory + "../problems/mnist/train-images.idx3-ubyte"};
        const idx_file_t train_labels{binary_directory + "../problems/mnist/train-labels.idx1-ubyte"};

        const image_t train_image{train_images, train_labels};

        network_stats_t stats;

        dlib::dnn_trainer trainer(network);
        trainer.set_learning_rate(0.01);
        trainer.set_min_learning_rate(0.00001);
        trainer.set_mini_batch_size(batch_size);
        trainer.set_max_num_epochs(100);
        trainer.set_iterations_without_progress_threshold(2000);
        trainer.be_verbose();

        trainer.set_synchronization_file("mnist_sync_" + ident, std::chrono::seconds(20));

        blt::size_t epochs = 0;
        if (last_epoch > 0)
            epochs = static_cast<blt::size_t>(last_epoch);
        blt::ptrdiff_t epoch_pos = 0;
        for (; epochs < trainer.get_max_num_epochs() && trainer.get_learning_rate() >= trainer.get_min_learning_rate(); epochs++)
        {
            auto& data = train_image.get_image_data();
            auto& labels = train_image.get_image_labels();
            for (; epoch_pos < data.size() && trainer.get_learning_rate() >= trainer.get_min_learning_rate(); epoch_pos += trainer.
                   get_mini_batch_size())
            {
                auto begin = epoch_pos;
                auto end = std::min(epoch_pos + trainer.get_mini_batch_size(), data.size());

                if (end - begin <= 0)
                    break;

                if (learn_flag)
                    trainer.set_learning_rate(trainer.get_learning_rate() / 10);

                trainer.train_one_step(train_image.get_image_data().begin() + begin,
                                       data.begin() + end, labels.begin() + begin);
            }
            epoch_pos = 0;
            BLT_TRACE("Trained an epoch (%ld/%ld)   learn rate %lf    average loss %lf", epochs, trainer.get_max_num_epochs(),
                      trainer.get_learning_rate(), trainer.get_average_loss());

            // sync and test
            trainer.get_net(dlib::force_flush_to_disk::no);
            network.clean();

            epoch_stats_t epoch_stats{};
            epoch_stats.test_results = test_batch(network, train_image.get_image_data().begin(), train_image.get_image_data().end(),
                                                  train_image.get_image_labels().begin());
            epoch_stats.average_loss = trainer.get_average_loss();
            epoch_stats.learn_rate = trainer.get_learning_rate();

            BLT_TRACE("\t\tHits: %lu\tMisses: %lu\tAccuracy: %lf", epoch_stats.test_results.hits, epoch_stats.test_results.misses,
                      epoch_stats.test_results.hits / static_cast<double>(epoch_stats.test_results.hits + epoch_stats.test_results.misses));

            stats.epoch_stats.push_back(epoch_stats);
            network.clean();
            if (break_flag)
            {
                break_flag = false;
                last_epoch = epochs + 1;
                break;
            }
            // dlib::serialize("mnist_network_" + ident + ".dat") << network;
        }

        BLT_INFO("Finished Training");

        // sync
        trainer.get_net();
        network.clean();

        // trainer.train(train_image.get_image_data(), train_image.get_image_labels());
        dlib::serialize("mnist_network_" + ident + ".dat") << network;

        auto test_results = test_batch(network, train_image.get_image_data().begin(), train_image.get_image_data().end(),
                                       train_image.get_image_labels().begin());

        BLT_DEBUG("Training hits: %lu", test_results.hits);
        BLT_DEBUG("Training misses: %lu", test_results.misses);
        BLT_DEBUG("Training accuracy: %lf", test_results.hits / static_cast<double>(test_results.hits + test_results.misses));

        return stats;
    }

    template <typename NetworkType>
    NetworkType load_network(const std::string& ident)
    {
        NetworkType network{};
        dlib::deserialize("mnist_network_" + ident + ".dat") >> network;
        return network;
    }

    template <typename NetworkType>
    std::pair<network_average_stats_t, batch_stats_t> run_network_tests(std::string path, const std::string& ident, const blt::i32 runs,
                                                                        const bool restore)
    {
        path += ("/" + ident + "/");
        std::filesystem::create_directories(path);
        std::filesystem::current_path(path);

        network_average_stats_t stats{};
        std::vector<batch_stats_t> test_stats;

        blt::i32 i = 0;
        if (std::filesystem::exists(path + "/state.bin"))
        {
            std::ifstream state{path + "/state.bin", std::ios::binary | std::ios::in};
            if (!state.is_open())
            {
                BLT_ERROR("Failed to open state file!");
                std::exit(-1);
            }

            state >> i;
            state.ignore();
            blt::i64 load_epoch = 0;
            state >> load_epoch;
            state.ignore();
            last_epoch = load_epoch;
            state >> stats;
            state.ignore();
            blt::size_t test_stats_size = 0;
            state >> test_stats_size;
            state.ignore();
            for (blt::size_t _ = 0; _ < test_stats_size; _++)
            {
                test_stats.emplace_back();
                state >> test_stats.back();
                state.ignore();
            }

            BLT_TRACE("Restoring at run %d with epoch %ld", i, load_epoch);
            BLT_TRACE("\tRestored state size %lu", stats.run_stats.size());
            BLT_TRACE("\tRestored test size %lu", test_stats_size);
        }

        blt::i64 last_epoch_save = last_epoch;
        for (; i < runs; i++)
        {
            if (stop_flag)
            {
                BLT_TRACE("Stopping!");
                break;
            }
            BLT_TRACE("Starting run %d", i);
            auto local_ident = ident + std::to_string(i);
            NetworkType network{};
            if (restore)
                try
                {
                    network = load_network<NetworkType>(local_ident);
                }
                catch (dlib::serialization_error&)
                {
                    goto train_label;
                }
            else
            {
                train_label:
                auto stat = train_network(local_ident, network);
                if (last_epoch_save > 0)
                {
                    // add in all the new epochs
                    auto& vec = stats.run_stats.back();
                    vec.epoch_stats.insert(vec.epoch_stats.end(), stat.epoch_stats.begin(), stat.epoch_stats.end());
                } else
                    stats += stat;
            }
            last_epoch_save = last_epoch;
            last_epoch = -1;
            test_stats.push_back(test_network(network));
        }

        batch_stats_t average;
        for (const auto& v : test_stats)
            average += v;
        average /= runs;

        std::ofstream state{path + "/state.bin", std::ios::binary | std::ios::out};
        if (!state.is_open())
        {
            BLT_ERROR("Failed to open state file!");
            std::exit(-1);
        }

        // user can skip this if required.
        state << std::max(i - 1, 0);
        state << '\n';
        state << last_epoch_save;
        state << '\n';
        state << stats;
        state << '\n';
        state << static_cast<blt::size_t>(std::max(static_cast<blt::i64>(test_stats.size()) - 1, 0l));
        state << '\n';
        if (!test_stats.empty())
        {
            // the last test stat will be recalculated on restore. keeping it is an error.
            for (const auto& v : blt::iterate(test_stats).take(test_stats.size() - 1))
            {
                state << v;
                state << '\n';
            }
        }

        return {stats, average};
    }

    auto run_deep_learning_tests(const std::string& path, const blt::i32 runs, const bool restore)
    {
        using namespace dlib;
        using net_type_dl = loss_multiclass_log<
            fc<10,
               relu<fc<84,
                       relu<fc<120,
                               max_pool<2, 2, 2, 2, relu<con<16, 5, 5, 1, 1,
                                                             max_pool<2, 2, 2, 2, relu<con<6, 5, 5, 1, 1,
                                                                                           input<matrix<blt::u8>>>>>>>>>>>>>>;
        BLT_TRACE("Running deep learning tests");
        return run_network_tests<net_type_dl>(path, "deep_learning", runs, restore);
    }

    auto run_feed_forward_tests(const std::string& path, const blt::i32 runs, const bool restore)
    {
        using namespace dlib;

        using net_type_ff = loss_multiclass_log<
            fc<10,
               relu<fc<84,
                       relu<fc<120,
                               input<matrix<blt::u8>>>>>>>>;

        BLT_TRACE("Running feed forward tests");
        return run_network_tests<net_type_ff>(path, "feed_forward", runs, restore);
    }

    void run_mnist(const int argc, const char** argv)
    {
        binary_directory = std::filesystem::current_path();
        blt::size_t pos = 0;
        if (!blt::string::ends_with(binary_directory, '/'))
        {
            pos = binary_directory.find_last_of('/');
            binary_directory += '/';
        }
        else
            pos = binary_directory.substr(0, binary_directory.size() - 1).find_last_of('/');
        python_dual_stacked_graph_program = binary_directory.substr(0, pos) + "/graph.py";
        BLT_DEBUG(binary_directory);
        BLT_DEBUG(python_dual_stacked_graph_program);
        BLT_DEBUG("Running with batch size %d", batch_size);

        BLT_DEBUG("Installing Signal Handlers");
        if (std::signal(SIGINT, [](int)
        {
            BLT_INFO("Stopping current training");
            break_flag = true;
        }) == SIG_ERR)
        {
            BLT_ERROR("Failed to replace SIGINT");
        }
        if (std::signal(SIGQUIT, [](int)
        {
            BLT_INFO("Exiting Program");
            stop_flag = true;
            break_flag = true;
        }) == SIG_ERR)
        {
            BLT_ERROR("Failed to replace SIGQUIT");
        }
        if (std::signal(SIGUSR1, [](int)
        {
            BLT_INFO("Decreasing Learn Rate for current training");
            learn_flag = true;
        }) == SIG_ERR)
        {
            BLT_ERROR("Failed to replace SIGUSR1");
        }
        if (std::signal(SIGUSR2, [](int)
        {
            BLT_INFO("Exiting Program");
            stop_flag = true;
            break_flag = true;
        }) == SIG_ERR)
        {
            BLT_ERROR("Failed to replace SIGUSR2");
        }

        using namespace dlib;

        blt::arg_parse parser{};
        parser.addArgument(
            blt::arg_builder{"-r", "--restore"}.setAction(blt::arg_action_t::STORE_TRUE).setDefault(false).setHelp(
                "Restores from last save").build());
        parser.addArgument(blt::arg_builder{"-t", "--runs"}.setHelp("Number of runs to perform [default: 10]").setDefault("10").build());
        parser.addArgument(
            blt::arg_builder{"-p", "--python"}.setHelp("Only run the python scripts").setAction(blt::arg_action_t::STORE_TRUE).setDefault(false).
                                               build());
        parser.addArgument(
            blt::arg_builder{"network"}.setDefault(std::to_string(blt::system::getCurrentTimeMilliseconds())).setHelp("location of network files").
                                        build());

        auto args = parser.parse_args(argc, argv);

        const auto runs = std::stoi(args.get<std::string>("runs"));
        const auto restore = args.get<bool>("restore");
        auto path = binary_directory + args.get<std::string>("network");

        auto [deep_stats, deep_tests] = run_deep_learning_tests(path, runs, restore);
        auto [forward_stats, forward_tests] = run_feed_forward_tests(path, runs, restore);

        auto average_forward_size = forward_stats.average_size();
        auto average_deep_size = deep_stats.average_size();

        {
            std::ofstream test_results_f{path + "/test_results_table.txt"};
            test_results_f << "\\begin{figure}" << std::endl;
            test_results_f << "\t\\begin{tabular}{|c|c|c|c|}" << std::endl;
            test_results_f << "\t\t\\hline" << std::endl;
            test_results_f << "\t\tTest & Correct & Incorrect & Accuracy (\\%) \\\\" << std::endl;
            test_results_f << "\t\t\\hline" << std::endl;
            auto test_accuracy = forward_tests.hits / static_cast<double>(forward_tests.hits + forward_tests.misses) * 100;
            test_results_f << "\t\tFeed-Forward & " << forward_tests.hits << " & " << forward_tests.misses << " & " << std::setprecision(2) <<
                test_accuracy << "\\\\" << std::endl;
            test_accuracy = deep_tests.hits / static_cast<double>(deep_tests.hits + deep_tests.misses) * 100;
            test_results_f << "\t\tDeep Learning & " << deep_tests.hits << " & " << deep_tests.misses << " & " << std::setprecision(2) <<
                test_accuracy << "\\\\" << std::endl;
            test_results_f << "\t\\end{tabular}" << std::endl;
            test_results_f << "\\end{figure}" << std::endl;

            const auto [forward_epoch_stats] = forward_stats.average_stats();
            std::ofstream train_forward{path + "/forward_train_results.csv"};
            train_forward << "Epoch,Loss" << std::endl;
            for (const auto& [i, v] : blt::enumerate(forward_epoch_stats))
                train_forward << i << ',' << v.average_loss << std::endl;

            const auto [deep_epoch_stats] = deep_stats.average_stats();
            std::ofstream train_deep{path + "/deep_train_results.csv"};
            train_deep << "Epoch,Loss" << std::endl;
            for (const auto& [i, v] : blt::enumerate(deep_epoch_stats))
                train_deep << i << ',' << v.average_loss << std::endl;

            std::ofstream average_epochs{path + "/average_epochs.txt"};
            average_epochs << average_forward_size << "," << average_deep_size << std::endl;
        }

        BLT_INFO("Running python!");
        run_python_line_graph("Feed-Forward vs Deep Learning, Average Loss over Epochs", "epochs.png", path + "/forward_train_results.csv",
                              path + "/deep_train_results.csv", average_forward_size, average_deep_size);

        // net_type_dl test_net;
        // const auto stats = train_network("dl_nn", test_net);
        // std::ofstream out_file{"dl_nn.csv"};
        // out_file << stats;

        // test_net = load_network<net_type_dl>("dl_nn");

        // test_network(test_net);
    }
}
