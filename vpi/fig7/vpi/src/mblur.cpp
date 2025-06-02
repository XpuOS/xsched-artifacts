#include "mblur.h"

MultiPvaBlurRunner::MultiPvaBlurRunner(const std::string &video_path_in, size_t num)
{
    for (size_t i = 0; i < num; ++i) {
        runners_.emplace_back(video_path_in);
        start_.emplace_back(false);
        done_.emplace_back(false);
        exit_.emplace_back(false);
    }
    for (size_t i = 0; i < num; ++i) {
        threads_.emplace_back([this, i]() {
            while (!exit_[i]) {
                std::unique_lock<std::mutex> lock(start_mtx_);
                while (!start_[i]) start_cv_.wait(lock);
                start_[i] = false;
                lock.unlock();

                if (exit_[i]) break;
                runners_[i].Enqueue(cmd_cnt_, sync_);
                runners_[i].Sync();

                done_mtx_.lock();
                done_[i] = true;
                done_mtx_.unlock();
                done_cv_.notify_all();
            }
        });
    }
}

void MultiPvaBlurRunner::Init()
{
    for (auto &runner : runners_) runner.Init();
}

void MultiPvaBlurRunner::Final()
{
    for (size_t i = 0; i < exit_.size(); ++i) exit_[i] = true;
    start_mtx_.lock();
    for (size_t i = 0; i < start_.size(); ++i) start_[i] = true;
    start_mtx_.unlock();
    start_cv_.notify_all();
    for (auto &thread : threads_) thread.join();
    for (auto &runner : runners_) runner.Final();
}

void MultiPvaBlurRunner::Execute(size_t cmd_cnt, bool sync)
{
    cmd_cnt_ = cmd_cnt;
    sync_ = sync;

    done_mtx_.lock();
    for (size_t i = 0; i < done_.size(); ++i) done_[i] = false;
    done_mtx_.unlock();

    start_mtx_.lock();
    for (size_t i = 0; i < start_.size(); ++i) start_[i] = true;
    start_mtx_.unlock();
    start_cv_.notify_all();

    std::unique_lock<std::mutex> lock(done_mtx_);
    done_cv_.wait(lock, [this]() {
        for (size_t i = 0; i < done_.size(); ++i) {
            if (!done_[i]) return false;
        }
        return true;
    });
    lock.unlock();
}
