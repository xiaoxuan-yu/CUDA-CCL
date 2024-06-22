/*
 * A timer class to measure the time taken by the algorithm
 * Xiaoxuan, Yu, parcoii2024
 */
#pragma once
#include <chrono>

class Timer
{
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time, temp_start_time, temp_end_time;

public:
    double duration;
    Timer()
    {
        start_time = std::chrono::high_resolution_clock::now();
        duration = 0.0;
        temp_start_time = start_time;
    }

    void pause()
    {
        temp_end_time = std::chrono::high_resolution_clock::now();
        this->duration += duration_in_microseconds(temp_start_time, temp_end_time);
    }

    void resume()
    {
        temp_start_time = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        end_time = std::chrono::high_resolution_clock::now();
        this->duration += duration_in_microseconds(temp_start_time, end_time);
    }

    static double duration_in_microseconds(std::chrono::time_point<std::chrono::high_resolution_clock> start, std::chrono::time_point<std::chrono::high_resolution_clock> end)
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e3;
    }
};

// overload the + operator to add two timers
inline Timer operator+(const Timer& t1, const Timer& t2)
{
    Timer t;
    t.duration = t1.duration + t2.duration;
    return t;
}

inline Timer operator/(const Timer& t, int n)
{
    Timer t_;
    t_.duration = t.duration / n;
    return t_;
}