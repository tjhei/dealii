// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


#include <taskflow/taskflow.hpp>

namespace taskflow_v1
{
  template <typename Worker,
            typename Copier,
            typename Iterator,
            typename ScratchData,
            typename CopyData>
  void
  run(const Iterator &                         begin,
      const typename identity<Iterator>::type &end,
      Worker                                   worker,
      Copier                                   copier,
      const ScratchData &                      sample_scratch_data,
      const CopyData &                         sample_copy_data,
      const unsigned int queue_length = 2 * MultithreadInfo::n_threads(),
      const unsigned int chunk_size   = 8)
  {
    if (MultithreadInfo::n_threads() == 1)
      {
        // need to copy the sample since it is marked const
        ScratchData scratch_data = sample_scratch_data;
        CopyData    copy_data    = sample_copy_data; // NOLINT

        for (Iterator i = begin; i != end; ++i)
          {
            // need to check if the function is not the zero function. To
            // check zero-ness, create a C++ function out of it and check that
            if (static_cast<const std::function<
                  void(const Iterator &, ScratchData &, CopyData &)> &>(worker))
              worker(i, scratch_data, copy_data);
            if (static_cast<const std::function<void(const CopyData &)> &>(
                  copier))
              copier(copy_data);
          }

        return;
      }

    tf::Executor &executor = MultithreadInfo::get_taskflow_executor();
    tf::Taskflow  taskflow;

    ScratchData scratch_data = sample_scratch_data;
    CopyData    copy_data    = sample_copy_data; // NOLINT

    tf::Task last_copier;

    std::vector<std::unique_ptr<CopyData>> copy_datas;

    unsigned int idx = 0;
    for (Iterator i = begin; i != end; ++i, ++idx)
      {
        copy_datas.emplace_back();

        auto worker_task = taskflow
                             .emplace([it = i,
                                       idx,
                                       &sample_scratch_data,
                                       &copy_datas,
                                       &sample_copy_data,
                                       &worker]() {
                               // std::cout << "worker " << idx << std::endl;
                               ScratchData scratch = sample_scratch_data;
                               auto &      copy    = copy_datas[idx];
                               copy =
                                 std::make_unique<CopyData>(sample_copy_data);

                               worker(it, scratch, *copy.get());
                             })
                             .name("worker");

        tf::Task copier_task = taskflow
                                 .emplace([idx, &copy_datas, &copier]() {
                                   copier(*copy_datas[idx].get());
                                   copy_datas[idx].reset();
                                 })
                                 .name("copy");

        worker_task.precede(copier_task);

        if (!last_copier.empty())
          last_copier.precede(copier_task);
        last_copier = copier_task;
      }

    executor.run(taskflow).wait();
    if (false)
      {
        std::ofstream f("graph.dia");
        taskflow.dump(f);
        f.close();
      }
  }


  template <typename MainClass,
            typename Iterator,
            typename ScratchData,
            typename CopyData>
  void
  run(const Iterator &                         begin,
      const typename identity<Iterator>::type &end,
      MainClass &                              main_object,
      void (MainClass::*worker)(const Iterator &, ScratchData &, CopyData &),
      void (MainClass::*copier)(const CopyData &),
      const ScratchData &sample_scratch_data,
      const CopyData &   sample_copy_data,
      const unsigned int queue_length = 2 * MultithreadInfo::n_threads(),
      const unsigned int chunk_size   = 8)
  {
    // forward to the other function
    run(begin,
        end,
        [&main_object, worker](const Iterator &iterator,
                               ScratchData &   scratch_data,
                               CopyData &      copy_data) {
          (main_object.*worker)(iterator, scratch_data, copy_data);
        },
        [&main_object, copier](const CopyData &copy_data) {
          (main_object.*copier)(copy_data);
        },
        sample_scratch_data,
        sample_copy_data,
        queue_length,
        chunk_size);
  }
} // namespace taskflow_v1


namespace taskflow_v4
{
  template <typename CopyData>
  class Chunk
  {
  public:
    Chunk(const unsigned int count, const CopyData &copy_data)
      : copy_datas(count, copy_data)
    {}
    std::vector<CopyData> copy_datas;
  };


  template <typename Worker,
            typename Copier,
            typename Iterator,
            typename ScratchData,
            typename CopyData>
  void
  run(const Iterator &                         begin,
      const typename identity<Iterator>::type &end,
      Worker                                   worker,
      Copier                                   copier,
      const ScratchData &                      sample_scratch_data,
      const CopyData &                         sample_copy_data,
      const unsigned int queue_length = 10 * MultithreadInfo::n_threads(),
      const unsigned int chunk_size   = 8)
  {
    if (MultithreadInfo::n_threads() == 1)
      {
        // need to copy the sample since it is marked const
        ScratchData scratch_data = sample_scratch_data;
        CopyData    copy_data    = sample_copy_data; // NOLINT

        for (Iterator i = begin; i != end; ++i)
          {
            // need to check if the function is not the zero function. To
            // check zero-ness, create a C++ function out of it and check that
            if (static_cast<const std::function<
                  void(const Iterator &, ScratchData &, CopyData &)> &>(worker))
              worker(i, scratch_data, copy_data);
            if (static_cast<const std::function<void(const CopyData &)> &>(
                  copier))
              copier(copy_data);
          }

        return;
      }

    tf::Executor &executor = MultithreadInfo::get_taskflow_executor();
    tf::Taskflow  taskflow;

    tf::Task last_copier = taskflow.placeholder();

    Threads::ThreadLocalStorage<std::unique_ptr<ScratchData>>
      thread_local_scratch;

    std::vector<std::unique_ptr<Chunk<CopyData>>> chunks;

    unsigned int idx             = 0;
    unsigned int remaining_items = std::distance(begin, end);

    const unsigned int real_chunk_size =
      (remaining_items / chunk_size < 3 * MultithreadInfo::n_threads()) ?
        1 :
        chunk_size;

    Iterator it = begin;
    while (it != end)
      {
        unsigned int count  = std::min(remaining_items, real_chunk_size);
        Iterator     middle = it;
        std::advance(middle, count);

        chunks.emplace_back();

        // this chunk works on [it,middle)
        auto worker_task =
          taskflow
            .emplace([it_begin = it,
                      it_end   = middle,
                      idx,
                      count,
                      &sample_scratch_data,
                      &thread_local_scratch,
                      &chunks,
                      &sample_copy_data,
                      &worker]() {
              auto &scratch_ptr = thread_local_scratch.get();
              if (!scratch_ptr.get())
                scratch_ptr =
                  std::make_unique<ScratchData>(sample_scratch_data);

              ScratchData &scratch = *scratch_ptr.get();
              chunks[idx] =
                std::make_unique<Chunk<CopyData>>(count, sample_copy_data);

              unsigned int counter = 0;
              for (Iterator it = it_begin; it != it_end; ++it, ++counter)
                {
                  worker(it, scratch, chunks[idx].get()->copy_datas[counter]);
                }
            })
            .name("work");

        tf::Task copier_task = taskflow
                                 .emplace([idx, &chunks, &copier]() mutable {
                                   auto chunk = chunks[idx].get();
                                   for (auto &cd : chunk->copy_datas)
                                     copier(cd);

                                   chunks[idx].reset();
                                 })
                                 .name("copy");

        worker_task.precede(copier_task);

        last_copier.precede(copier_task);
        last_copier = copier_task;

        it = middle;
        ++idx;
      }

    // debugging:

    executor.run(taskflow).wait();

#ifdef DEBUG
    std::cout << "done" << std::endl;
    std::ofstream f("graph.dia");
    taskflow.dump(f);
    f.close();
#endif
  }


  /**
   * WorkStream without colored iterators.
   *
   * Work in chunks of size @p chunk_size (only if we have enough items, otherwise 1) and
   * use a thread-local scratch object
   */
  template <typename MainClass,
            typename Iterator,
            typename ScratchData,
            typename CopyData>
  void
  run(const Iterator &                         begin,
      const typename identity<Iterator>::type &end,
      MainClass &                              main_object,
      void (MainClass::*worker)(const Iterator &, ScratchData &, CopyData &),
      void (MainClass::*copier)(const CopyData &),
      const ScratchData &sample_scratch_data,
      const CopyData &   sample_copy_data,
      const unsigned int queue_length = 2 * MultithreadInfo::n_threads(),
      const unsigned int chunk_size   = 8)
  {
    // forward to the other function
    run(begin,
        end,
        [&main_object, worker](const Iterator &iterator,
                               ScratchData &   scratch_data,
                               CopyData &      copy_data) {
          (main_object.*worker)(iterator, scratch_data, copy_data);
        },
        [&main_object, copier](const CopyData &copy_data) {
          (main_object.*copier)(copy_data);
        },
        sample_scratch_data,
        sample_copy_data,
        queue_length,
        chunk_size);
  }

  /**
   * Colored WorkStream version using taskflow
   *
   * We use a thread-local scratch object and create a single task for a chunk
   * of worker&copier tasks together. Within each color, all tasks are
   * independent. The @p chunk_size parameter determines the number of worker
   * to group into a single task (unless the number of tasks would be too
   * small).
   */
  template <typename Worker,
            typename Copier,
            typename Iterator,
            typename ScratchData,
            typename CopyData>
  void
  run(const std::vector<std::vector<Iterator>> &colored_iterators,
      Worker                                    worker,
      Copier                                    copier,
      const ScratchData &                       sample_scratch_data,
      const CopyData &                          sample_copy_data,
      const unsigned int                        chunk_size = 8)
  {
    if (MultithreadInfo::n_threads() == 1)
      WorkStream::internal::sequential::run(colored_iterators,
                                            worker,
                                            copier,
                                            sample_scratch_data,
                                            sample_copy_data);
    else
      {
        tf::Executor &executor = MultithreadInfo::get_taskflow_executor();

        Threads::ThreadLocalStorage<std::unique_ptr<ScratchData>>
          thread_local_scratch;

        for (unsigned int color = 0; color < colored_iterators.size(); ++color)
          if (colored_iterators[color].size() > 0)
            {
              // package worker&copier into a task each and schedule all of them
              // at the same time:
              tf::Taskflow taskflow;

              unsigned int remaining_items = colored_iterators[color].size();

              const unsigned int real_chunk_size =
                (remaining_items / chunk_size <
                 3 * MultithreadInfo::n_threads()) ?
                  1 :
                  chunk_size;

              auto it  = colored_iterators[color].begin();
              auto end = colored_iterators[color].end();

              while (it != end)
                {
                  unsigned int count =
                    std::min(remaining_items, real_chunk_size);
                  auto middle = it;
                  std::advance(middle, count);



                  auto worker_task =
                    taskflow
                      .emplace([it_begin = it,
                                it_end   = middle,
                                &sample_scratch_data,
                                &thread_local_scratch,
                                &sample_copy_data,
                                &worker,
                                &copier]() {
                        auto &scratch_ptr = thread_local_scratch.get();
                        if (!scratch_ptr.get())
                          scratch_ptr =
                            std::make_unique<ScratchData>(sample_scratch_data);

                        ScratchData &scratch = *scratch_ptr.get();

                        for (auto it = it_begin; it != it_end; ++it)
                          {
                            CopyData copy_data = sample_copy_data;
                            worker(*it, scratch, copy_data);
                            copier(copy_data);
                          }
                      })
                      .name("worker");

                  it = middle;
                  remaining_items -= count;
                }



              // make sure we finish all of them:
              executor.run(taskflow).wait();
            }
      }
  }


} // namespace taskflow_v4
