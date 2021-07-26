#pragma once

#include <chrono>
#include <future>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>

#include "batch_reader.h"
#include "translator.h"

namespace ctranslate2 {

  struct TranslationStats {
    size_t num_tokens = 0;
    size_t num_examples = 0;
    double total_time_in_ms = 0;
  };

  class BufferedTranslationWrapper;

  // TranslatorPool is the high-level class for running translations. It supports parallel
  // and asynchronous translations.
  class TranslatorPool {
  public:
    // num_translators (a.k.a. inter_threads) and num_threads_per_translator (a.k.a. intra_threads)
    // are forced to 1 when the translator is running on a CUDA device.
    TranslatorPool(size_t num_translators,
                   size_t num_threads_per_translator,
                   const std::string& model_dir,
                   const Device device = Device::CPU,
                   const int device_index = 0,
                   const ComputeType compute_type = ComputeType::DEFAULT);

    // Multi-device constructor.
    TranslatorPool(size_t num_translators_per_device,
                   size_t num_threads_per_translator,
                   const std::string& model_dir,
                   const Device device,
                   const std::vector<int>& device_indices,
                   const ComputeType compute_type = ComputeType::DEFAULT);

    ~TranslatorPool();

    std::vector<std::future<TranslationResult>>
    translate_batch_async(const std::vector<std::vector<std::string>>& source,
                          const TranslationOptions& options = TranslationOptions(),
                          const size_t max_batch_size = 0,
                          const BatchType batch_type = BatchType::Examples);
    std::vector<std::future<TranslationResult>>
    translate_batch_async(const std::vector<std::vector<std::string>>& source,
                          const std::vector<std::vector<std::string>>& target_prefix,
                          const TranslationOptions& options = TranslationOptions(),
                          const size_t max_batch_size = 0,
                          const BatchType batch_type = BatchType::Examples);

    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& source,
                    const TranslationOptions& options = TranslationOptions(),
                    const size_t max_batch_size = 0,
                    const BatchType batch_type = BatchType::Examples);
    std::vector<TranslationResult>
    translate_batch(const std::vector<std::vector<std::string>>& source,
                    const std::vector<std::vector<std::string>>& target_prefix,
                    const TranslationOptions& options = TranslationOptions(),
                    const size_t max_batch_size = 0,
                    const BatchType batch_type = BatchType::Examples);

    std::vector<std::future<ScoringResult>>
    score_batch_async(const std::vector<std::vector<std::string>>& source,
                      const std::vector<std::vector<std::string>>& target,
                      const size_t max_batch_size = 0,
                      const BatchType batch_type = BatchType::Examples);
    std::vector<ScoringResult>
    score_batch(const std::vector<std::vector<std::string>>& source,
                const std::vector<std::vector<std::string>>& target,
                const size_t max_batch_size = 0,
                const BatchType batch_type = BatchType::Examples);

    // Translate a stream.
    // The reader and writer functions do not need to be thread-safe.
    template <typename Reader, typename Writer>
    void consume_stream(std::istream& in,
                        std::ostream& out,
                        Reader& reader,
                        Writer& writer,
                        const TranslationOptions& options = TranslationOptions(),
                        size_t max_batch_size = 32,
                        size_t read_batch_size = 0,
                        BatchType batch_type = BatchType::Examples) {
      return consume_stream(in,
                            nullptr,
                            out,
                            reader,
                            nullptr,
                            writer,
                            options,
                            max_batch_size,
                            read_batch_size,
                            batch_type);
    }

    template <typename SourceReader, typename TargetReader, typename TargetWriter>
    void consume_stream(std::istream& source,
                        std::istream* target,
                        std::ostream& output,
                        SourceReader& source_reader,
                        TargetReader* target_reader,
                        TargetWriter& target_writer,
                        const TranslationOptions& options = TranslationOptions(),
                        size_t max_batch_size = 32,
                        size_t read_batch_size = 0,
                        BatchType batch_type = BatchType::Examples) {
      TranslateJobCreator job_creator(options);
      consume_stream(source,
                     target,
                     output,
                     source_reader,
                     target_reader,
                     target_writer,
                     job_creator,
                     max_batch_size,
                     read_batch_size,
                     batch_type);
    }

    // Translate a file.
    // These are wrappers around consume_stream that set the appropriate reader and writer.
    TranslationStats consume_text_file(const std::string& source_file,
                                       const std::string& output_file,
                                       const TranslationOptions& options = TranslationOptions(),
                                       size_t max_batch_size = 32,
                                       size_t read_batch_size = 0,
                                       BatchType batch_type = BatchType::Examples,
                                       bool with_scores = false,
                                       const std::string* target_file = nullptr);

    TranslationStats consume_text_file(std::istream& source,
                                       std::ostream& output,
                                       const TranslationOptions& options = TranslationOptions(),
                                       size_t max_batch_size = 32,
                                       size_t read_batch_size = 0,
                                       BatchType batch_type = BatchType::Examples,
                                       bool with_scores = false,
                                       std::istream* target = nullptr);

    template <typename Tokenizer, typename Detokenizer>
    TranslationStats consume_raw_text_file(const std::string& in_file,
                                           const std::string& out_file,
                                           Tokenizer& tokenizer,
                                           Detokenizer& detokenizer,
                                           const TranslationOptions& options = TranslationOptions(),
                                           const size_t max_batch_size = 32,
                                           const size_t read_batch_size = 0,
                                           const BatchType batch_type = BatchType::Examples,
                                           const bool with_scores = false) {
      std::ifstream in;
      open_input_file(in_file, in);
      std::ofstream out;
      open_output_file(out_file, out);
      return consume_raw_text_file(in,
                                   out,
                                   tokenizer,
                                   detokenizer,
                                   options,
                                   max_batch_size,
                                   read_batch_size,
                                   batch_type,
                                   with_scores);

    }

    template <typename Tokenizer, typename Detokenizer>
    TranslationStats consume_raw_text_file(std::istream& in,
                                           std::ostream& out,
                                           Tokenizer& tokenizer,
                                           Detokenizer& detokenizer,
                                           const TranslationOptions& options = TranslationOptions(),
                                           const size_t max_batch_size = 32,
                                           const size_t read_batch_size = 0,
                                           const BatchType batch_type = BatchType::Examples,
                                           const bool with_scores = false) {
      return consume_raw_text_file(in,
                                   nullptr,
                                   out,
                                   tokenizer,
                                   tokenizer,
                                   detokenizer,
                                   options,
                                   max_batch_size,
                                   read_batch_size,
                                   batch_type,
                                   with_scores);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats consume_raw_text_file(const std::string& source_file,
                                           const std::string* target_file,
                                           const std::string& output_file,
                                           SourceTokenizer& source_tokenizer,
                                           TargetTokenizer& target_tokenizer,
                                           TargetDetokenizer& detokenizer,
                                           const TranslationOptions& options = TranslationOptions(),
                                           const size_t max_batch_size = 32,
                                           const size_t read_batch_size = 0,
                                           const BatchType batch_type = BatchType::Examples,
                                           const bool with_scores = false) {
      std::ifstream source;
      open_input_file(source_file, source);
      std::ofstream output;
      open_output_file(output_file, output);

      std::unique_ptr<std::ifstream> target;
      if (target_file) {
        target = std::make_unique<std::ifstream>();
        open_input_file(*target_file, *target);
      }

      return consume_raw_text_file(source,
                                   target.get(),
                                   output,
                                   source_tokenizer,
                                   target_tokenizer,
                                   detokenizer,
                                   options,
                                   max_batch_size,
                                   read_batch_size,
                                   batch_type,
                                   with_scores);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats consume_raw_text_file(std::istream& source,
                                           std::istream* target,
                                           std::ostream& output,
                                           SourceTokenizer& source_tokenizer,
                                           TargetTokenizer& target_tokenizer,
                                           TargetDetokenizer& detokenizer,
                                           const TranslationOptions& options = TranslationOptions(),
                                           const size_t max_batch_size = 32,
                                           const size_t read_batch_size = 0,
                                           const BatchType batch_type = BatchType::Examples,
                                           const bool with_scores = false) {
      TranslationStats stats;

      TokensReader<SourceTokenizer> source_reader(source_tokenizer);
      TokensReader<TargetTokenizer> target_reader(target_tokenizer);

      auto writer = [&detokenizer, &stats, &with_scores](std::ostream& out,
                                                         const TranslationResult& result) {
        const auto& hypotheses = result.hypotheses;
        const auto& scores = result.scores;
        stats.num_examples += 1;
        stats.num_tokens += hypotheses[0].size();
        for (size_t n = 0; n < hypotheses.size(); ++n) {
          if (with_scores)
            out << (result.has_scores() ? scores[n] : 0) << " ||| ";
          out << detokenizer(hypotheses[n]) << '\n';
        }
      };

      const auto t1 = std::chrono::high_resolution_clock::now();

      consume_stream(source,
                     target,
                     output,
                     source_reader,
                     &target_reader,
                     writer,
                     options,
                     max_batch_size,
                     read_batch_size,
                     batch_type);

      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

    // Score a stream.
    // The reader and writer functions do not need to be thread-safe.
    template <typename SourceReader, typename TargetReader, typename TargetWriter>
    void score_stream(std::istream& source,
                      std::istream& target,
                      std::ostream& output,
                      SourceReader& source_reader,
                      TargetReader& target_reader,
                      TargetWriter& target_writer,
                      size_t max_batch_size = 32,
                      size_t read_batch_size = 0,
                      BatchType batch_type = BatchType::Examples) {
      ScoreJobCreator job_creator;
      consume_stream(source,
                     &target,
                     output,
                     source_reader,
                     &target_reader,
                     target_writer,
                     job_creator,
                     max_batch_size,
                     read_batch_size,
                     batch_type);
    }

    // Score a file.
    // These are wrappers around score_stream that set the appropriate reader and writer.
    TranslationStats score_text_file(const std::string& source_file,
                                     const std::string& target_file,
                                     const std::string& output_file,
                                     size_t max_batch_size = 32,
                                     size_t read_batch_size = 0,
                                     BatchType batch_type = BatchType::Examples);
    TranslationStats score_text_file(std::istream& source,
                                     std::istream& target,
                                     std::ostream& output,
                                     size_t max_batch_size = 32,
                                     size_t read_batch_size = 0,
                                     BatchType batch_type = BatchType::Examples);

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats score_raw_text_file(const std::string& source_file,
                                         const std::string& target_file,
                                         const std::string& output_file,
                                         SourceTokenizer& source_tokenizer,
                                         TargetTokenizer& target_tokenizer,
                                         TargetDetokenizer& target_detokenizer,
                                         const size_t max_batch_size = 32,
                                         const size_t read_batch_size = 0,
                                         const BatchType batch_type = BatchType::Examples) {
      std::ifstream source;
      open_input_file(source_file, source);
      std::ifstream target;
      open_input_file(target_file, target);
      std::ofstream output;
      open_output_file(output_file, output);
      return score_raw_text_file(source,
                                 target,
                                 output,
                                 source_tokenizer,
                                 target_tokenizer,
                                 target_detokenizer,
                                 max_batch_size,
                                 read_batch_size,
                                 batch_type);
    }

    template <typename SourceTokenizer, typename TargetTokenizer, typename TargetDetokenizer>
    TranslationStats score_raw_text_file(std::istream& source,
                                         std::istream& target,
                                         std::ostream& output,
                                         SourceTokenizer& source_tokenizer,
                                         TargetTokenizer& target_tokenizer,
                                         TargetDetokenizer& target_detokenizer,
                                         const size_t max_batch_size = 32,
                                         const size_t read_batch_size = 0,
                                         const BatchType batch_type = BatchType::Examples) {
      TokensReader<SourceTokenizer> source_reader(source_tokenizer);
      TokensReader<TargetTokenizer> target_reader(target_tokenizer);
      TranslationStats stats;

      auto writer = [&target_detokenizer, &stats](std::ostream& out, const ScoringResult& result) {
        stats.num_examples += 1;
        stats.num_tokens += result.tokens_score.size();
        out << result.normalized_score() << " ||| " << target_detokenizer(result.tokens) << '\n';
      };

      const auto t1 = std::chrono::high_resolution_clock::now();
      score_stream(source,
                   target,
                   output,
                   source_reader,
                   target_reader,
                   writer,
                   max_batch_size,
                   read_batch_size,
                   batch_type);
      const auto t2 = std::chrono::high_resolution_clock::now();
      stats.total_time_in_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
        t2 - t1).count();
      return stats;
    }

    size_t num_queued_batches();
    size_t num_translators() const;
    const std::vector<Translator>& get_translators() const;

  private:
    friend class BufferedTranslationWrapper;

    class Job {
    public:
      virtual ~Job() = default;
      virtual void run(Translator& translator) = 0;
    };

    // Base class for consuming job results.
    template <typename Result>
    class JobResultConsumer {
    public:
      JobResultConsumer(size_t num_results)
        : _promises(num_results)
      {
      }

      JobResultConsumer(std::vector<std::promise<Result>> promises)
        : _promises(std::move(promises))
      {
      }

      std::vector<std::future<Result>> get_futures() {
        std::vector<std::future<Result>> futures;
        futures.reserve(_promises.size());
        for (auto& promise : _promises)
          futures.emplace_back(promise.get_future());
        return futures;
      }

      void set_result(size_t index, Result result) {
        _promises[index].set_value(std::move(result));
      }

      void set_exception(size_t index, std::exception_ptr exception) {
        _promises[index].set_exception(exception);
      }

    private:
      std::vector<std::promise<Result>> _promises;
    };

    template <typename Result>
    class BatchJob : public Job {
    public:
      BatchJob(Batch batch, std::shared_ptr<JobResultConsumer<Result>> consumer)
        : _batch(std::move(batch))
        , _consumer(std::move(consumer))
      {
      }

      void run(Translator& translator) override {
        std::vector<Result> results;
        std::exception_ptr exception;

        try {
          results = get_results(translator, _batch);
        } catch (...) {
          exception = std::current_exception();
        }

        for (size_t i = 0; i < _batch.source.size(); ++i) {
          const size_t index = (_batch.example_index.empty() ? i : _batch.example_index[i]);
          if (exception)
            _consumer->set_exception(index, exception);
          else
            _consumer->set_result(index, std::move(results[i]));
        }
      }

    protected:
      virtual std::vector<Result>
      get_results(Translator& translator, const Batch& batch) const = 0;

    private:
      const Batch _batch;
      const std::shared_ptr<JobResultConsumer<Result>> _consumer;
    };

    class TranslateJob : public BatchJob<TranslationResult> {
    public:
      TranslateJob(Batch batch,
                   TranslationOptions options,
                   std::shared_ptr<JobResultConsumer<TranslationResult>> consumer);

    protected:
      std::vector<TranslationResult>
      get_results(Translator& translator, const Batch& batch) const override;

    private:
      const TranslationOptions _options;
    };

    class ScoreJob : public BatchJob<ScoringResult> {
    public:
      ScoreJob(Batch batch, std::shared_ptr<JobResultConsumer<ScoringResult>> consumer);

    protected:
      std::vector<ScoringResult>
      get_results(Translator& translator, const Batch& batch) const override;
    };

    template <typename Result>
    class JobCreator {
    public:
      virtual ~JobCreator() = default;

      std::vector<std::future<Result>> post(TranslatorPool& pool,
                                            const std::vector<std::vector<std::string>>& source,
                                            const std::vector<std::vector<std::string>>& target,
                                            size_t max_batch_size,
                                            BatchType batch_type,
                                            bool throttle) const {
        if (source.empty())
          return {};

        auto batches = create_batches(source, target, max_batch_size, batch_type);
        auto consumer = create_consumer(source, target);
        auto futures = consumer->get_futures();

        for (auto& batch : batches)
          pool.post_job(create_job(std::move(batch), consumer), throttle);

        return futures;
      }

    protected:
      virtual std::shared_ptr<JobResultConsumer<Result>>
      create_consumer(const std::vector<std::vector<std::string>>& source,
                      const std::vector<std::vector<std::string>>& target) const {
        (void)target;
        return std::make_shared<JobResultConsumer<Result>>(source.size());
      }

      virtual std::vector<Batch>
      create_batches(const std::vector<std::vector<std::string>>& source,
                     const std::vector<std::vector<std::string>>& target,
                     size_t max_batch_size,
                     BatchType batch_type) const {
        return rebatch_input(source, target, max_batch_size, batch_type, /*filter_empty=*/false);
      }

      virtual std::unique_ptr<Job>
      create_job(Batch batch, std::shared_ptr<JobResultConsumer<Result>> consumer) const = 0;
    };

    class TranslateJobCreator : public JobCreator<TranslationResult> {
    public:
      TranslateJobCreator(TranslationOptions options)
        : _options(std::move(options))
      {
        _options.validate();
      }

    protected:
      std::shared_ptr<JobResultConsumer<TranslationResult>>
      create_consumer(const std::vector<std::vector<std::string>>& source,
                      const std::vector<std::vector<std::string>>& target) const override {
        auto consumer = JobCreator<TranslationResult>::create_consumer(source, target);

        // Directly set an empty result for empty inputs.
        for (size_t i = 0; i < source.size(); ++i) {
          if (source[i].empty()) {
            consumer->set_result(i, TranslationResult(_options.num_hypotheses,
                                                      _options.return_attention,
                                                      _options.return_scores));
          }
        }

        return consumer;
      }

      std::vector<Batch>
      create_batches(const std::vector<std::vector<std::string>>& source,
                     const std::vector<std::vector<std::string>>& target,
                     size_t max_batch_size,
                     BatchType batch_type) const override {
        if (!_options.support_batch_translation()) {
          max_batch_size = 1;
          batch_type = BatchType::Examples;
        }
        return rebatch_input(source, target, max_batch_size, batch_type);
      }

      std::unique_ptr<Job>
      create_job(Batch batch,
                 std::shared_ptr<JobResultConsumer<TranslationResult>> consumer) const override {
        return std::make_unique<TranslateJob>(std::move(batch), _options, std::move(consumer));
      }

    private:
      const TranslationOptions _options;
    };

    class ScoreJobCreator : public JobCreator<ScoringResult> {
    protected:
      std::unique_ptr<Job>
      create_job(Batch batch,
                 std::shared_ptr<JobResultConsumer<ScoringResult>> consumer) const override {
        return std::make_unique<ScoreJob>(std::move(batch), std::move(consumer));
      }
    };

    template <typename SourceReader,
              typename TargetReader,
              typename TargetWriter,
              typename Result>
    void consume_stream(std::istream& source,
                        std::istream* target,
                        std::ostream& output,
                        SourceReader& source_reader,
                        TargetReader* target_reader,
                        TargetWriter& target_writer,
                        const JobCreator<Result>& job_creator,
                        size_t max_batch_size,
                        size_t read_batch_size,
                        BatchType batch_type) {
      std::queue<std::future<Result>> results;

      auto pop_results = [&results, &output, &target_writer](bool blocking) {
        static const auto zero_sec = std::chrono::seconds(0);
        while (!results.empty()
               && (blocking
                   || results.front().wait_for(zero_sec) == std::future_status::ready)) {
          target_writer(output, results.front().get());
          results.pop();
        }
      };

      ParallelBatchReader batch_reader;
      batch_reader.add(std::make_unique<StreamReader<SourceReader>>(source, source_reader));
      if (target) {
        batch_reader.add(std::make_unique<StreamReader<TargetReader>>(*target, *target_reader));
      }

      if (read_batch_size == 0)
        read_batch_size = max_batch_size * 16;

      while (true) {
        auto batch = batch_reader.get_next(read_batch_size, batch_type);
        if (batch[0].empty())
          break;
        auto futures = job_creator.post(*this,
                                        batch[0],
                                        target ? batch[1] : std::vector<std::vector<std::string>>(),
                                        max_batch_size,
                                        batch_type,
                                        /*throttle=*/true);
        for (auto& future : futures)
          results.emplace(std::move(future));

        pop_results(/*blocking=*/false);
      }

      pop_results(/*blocking=*/true);
      output.flush();
    }

    void create_translators(size_t num_translators_per_device,
                            size_t num_threads_per_translator,
                            const std::string& model_dir,
                            const Device device,
                            std::vector<int> device_indices,
                            const ComputeType compute_type);

    // With throttle=true it will block if there is already too much work pending.
    void post_job(std::unique_ptr<Job> job, bool throttle);
    void work_loop(Translator& translator, size_t num_threads);

    void open_input_file(const std::string& file, std::ifstream& stream) const;
    void open_output_file(const std::string& file, std::ofstream& stream) const;

    std::condition_variable _can_add_job;
    std::condition_variable _can_get_job;
    std::queue<std::unique_ptr<Job>> _work;
    std::vector<std::thread> _workers;
    std::vector<Translator> _translators;
    std::mutex _mutex;
    bool _request_end = false;

    template <typename Tokenizer>
    class TokensReader {
    public:
      TokensReader(Tokenizer& tokenizer)
        : _tokenizer(tokenizer)
      {
      }

      bool operator()(std::istream& in, std::vector<std::string>& tokens) {
        std::string line;
        if (!std::getline(in, line))
          return false;
        tokens = _tokenizer(line);
        return true;
      }

    private:
      Tokenizer& _tokenizer;
    };
  };

}
