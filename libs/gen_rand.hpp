/*  Written in 2016 by Andy Gainey (againey@experilous.com)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#pragma once

#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <random>
#include <future>

using std::uint32_t;
using std::uint64_t;

// OS-specific functions; add your own for other environments/platforms.

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#define NOINLINE __declspec(noinline)

uint64_t get_current_tick()
{
	uint64_t tick;
	QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&tick));
	return tick;
}

uint64_t get_tick_count(double seconds)
{
	uint64_t freq;
	QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&freq));
	return static_cast<uint64_t>(static_cast<double>(freq) * seconds);
}

void prepare_thread()
{
	HANDLE hProc = GetCurrentProcess();
	HANDLE hThread = GetCurrentThread();
	DWORD_PTR proc_affinity_mask, sys_affinity_mask;
	GetProcessAffinityMask(hProc, &proc_affinity_mask, &sys_affinity_mask);
	proc_affinity_mask = 0x1ULL << 63;
	while ((proc_affinity_mask & sys_affinity_mask) == 0ULL && proc_affinity_mask != 0ULL)
	{
		proc_affinity_mask = proc_affinity_mask >> 1;
	}
	if (proc_affinity_mask == 0ULL) proc_affinity_mask = sys_affinity_mask;
	SetProcessAffinityMask(hProc, proc_affinity_mask);
	SetThreadAffinityMask(hThread, proc_affinity_mask);

	SetPriorityClass(hProc, ABOVE_NORMAL_PRIORITY_CLASS);
	SetThreadPriority(hThread, THREAD_PRIORITY_ABOVE_NORMAL);
}

#endif

float as_float(uint32_t i)
{
	union
	{
		uint32_t i;
		float f;
	} pun = { i };
	return pun.f;
}

float as_float_cast(uint32_t i)
{
	union
	{
		float f;
		uint32_t i;
	} pun = { (float)i };
	pun.i -= 0x0C000000U;
	return pun.f;
}

double as_double(uint64_t i)
{
	union
	{
		uint64_t i;
		double f;
	} pun = { i };
	return pun.f;
}

double as_double_cast(uint64_t i)
{
	union
	{
		double f;
		uint64_t i;
	} pun = { (double)i };
	pun.i -= 0x0350000000000000ULL;
	return pun.f;
}

uint32_t as_int(float f)
{
	union
	{
		float f;
		uint32_t i;
	} pun = { f };
	return pun.i;
}

uint64_t as_int(double f)
{
	union
	{
		double f;
		uint64_t i;
	} pun = { f };
	return pun.i;
}

// http://xoroshiro.di.unimi.it/xoroshiro128plus.c
class xoroshiro_128_plus
{
private:
	uint64_t state0;
	uint64_t state1;

	static inline uint64_t rotl(const uint64_t x, int k)
	{
		return (x << k) | (x >> (64 - k));
	}

public:
	xoroshiro_128_plus() = default;
	xoroshiro_128_plus(uint64_t state0, uint64_t state1) : state0(state0), state1(state1) {}

	uint64_t next64()
	{
		const uint64_t s0 = state0;
		uint64_t s1 = state1;
		uint64_t r = s0 + s1;
		s1 ^= s0;
		state0 = rotl(s0, 55) ^ s1 ^ (s1 << 14);
		state1 = rotl(s1, 36);
		return r;
	}

	uint32_t next32()
	{
		const uint64_t s0 = state0;
		uint64_t s1 = state1;
		uint64_t r = s0 + s1;
		s1 ^= s0;
		state0 = rotl(s0, 55) ^ s1 ^ (s1 << 14);
		state1 = rotl(s1, 36);
		return static_cast<uint32_t>(r);
	}
};

// http://xoroshiro.di.unimi.it/xorshift128plus.c
class xorshift_128_plus
{
private:
	uint64_t state0;
	uint64_t state1;

public:
	xorshift_128_plus() = default;
	xorshift_128_plus(uint64_t state0, uint64_t state1) : state0(state0), state1(state1) {}

	uint64_t next64()
	{
		uint64_t x = state0;
		uint64_t y = state1;
		uint64_t r = x + y;
		state0 = y;
		x ^= x << 23;
		state1 = x ^ y ^ (x >> 18) ^ (y >> 5);
		return r;
	}

	uint32_t next32()
	{
		uint64_t x = state0;
		uint64_t y = state1;
		uint64_t r = x + y;
		state0 = y;
		x ^= x << 23;
		state1 = x ^ y ^ (x >> 18) ^ (y >> 5);
		return static_cast<uint32_t>(r);
	}
};

template <typename TRandom>
float rand_float_co(TRandom& random)
{
	return as_float(0x3F800000U | (random.next32() >> 9)) - 1.0f;
}

template <typename TRandom>
double rand_double_co(TRandom& random)
{
	return as_double(0x3FF0000000000000ULL | (random.next64() >> 12)) - 1.0;
}

template <typename TRandom>
float rand_float_oc(TRandom& random)
{
	return 2.0f - as_float(0x3F800000U | (random.next32() >> 9));
}

template <typename TRandom>
double rand_double_oc(TRandom& random)
{
	return 2.0 - as_double(0x3FF0000000000000ULL | (random.next64() >> 12));
}

template <typename TRandom>
float rand_float_oo(TRandom& random)
{
	uint32_t n;
	do
	{
		n = random.next32();
	} while (n <= 0x000001FFU); // If true, then the highest 23 bits must all be zero.
	return as_float(0x3F800000U | (n >> 9)) - 1.0f;
}

template <typename TRandom>
double rand_double_oo(TRandom& random)
{
	uint64_t n;
	do
	{
		n = random.next64();
	} while (n <= 0x0000000000000FFFULL); // If true, then the highest 52 bits must all be zero.
	return as_double(0x3FF0000000000000ULL | (n >> 12)) - 1.0;
}

// Force no-inline, because otherwise, the compiler might be inclined to inline this
// into the non-critical slow path of the calling function, which in turns makes that
// function too large to inline, hampering its fast path with an unnecessary function
// call indirection.
template <typename TRandom>
NOINLINE bool rand_probability(TRandom& random, uint32_t numerator, uint32_t denominator)
{
	uint32_t mask = denominator - 1U;
	mask |= mask >> 1;
	mask |= mask >> 2;
	mask |= mask >> 4;
	mask |= mask >> 8;
	mask |= mask >> 16;
	uint32_t n;
	do
	{
		n = random.next32() & mask;
	} while (n >= denominator);
	return n < numerator;
}

template <typename TRandom>
NOINLINE bool rand_probability(TRandom& random, uint64_t numerator, uint64_t denominator)
{
	uint64_t mask = denominator - 1ULL;
	mask |= mask >> 1;
	mask |= mask >> 2;
	mask |= mask >> 4;
	mask |= mask >> 8;
	mask |= mask >> 16;
	mask |= mask >> 32;
	uint64_t n;
	do
	{
		n = random.next64() & mask;
	} while (n >= denominator);
	return n < numerator;
}

template <typename TRandom>
float rand_float_cc(TRandom& random)
{
	uint32_t n = random.next32();
	// First check if the upper 9 bits are all set.
	// If not, short circuit to the else block.
	// If so, carry on with the messy check.
	if (n >= 0xFF800000U && rand_probability(random, 0x00000200U, 0x00800001U))
	{
		return 1.0f;
	}
	else
	{
		return as_float(0x3F800000U | (n & 0x007FFFFFU)) - 1.0f;
	}
}

template <typename TRandom>
double rand_double_cc(TRandom& random)
{
	uint64_t n = random.next64();
	// First check if the upper 12 bits are all set.
	// If not, short circuit to the else block.
	// If so, carry on with the messy check.
	if (n >= 0xFFF0000000000000ULL && rand_probability(random, 0x00001000ULL, 0x0010000000000001ULL))
	{
		return 1.0;
	}
	else
	{
		return as_double(0x3FF0000000000000ULL | (n & 0x000FFFFFFFFFFFFFULL)) - 1.0;
	}
}

template <typename TRandom>
float rand_float_co_cast(TRandom& random)
{
	auto n = random.next32() >> 8;
	return n != 0U ? as_float_cast(n) : 0.0f;
}

template <typename TRandom>
double rand_double_co_cast(TRandom& random)
{
	auto n = random.next64() >> 11;
	return n != 0ULL ? as_double_cast(n) : 0.0;
}

template <typename TRandom>
float rand_float_oc_cast(TRandom& random)
{
	auto n = random.next32() >> 8;
	return n != 0U ? as_float_cast(n) : 1.0f;
}

template <typename TRandom>
double rand_double_oc_cast(TRandom& random)
{
	auto n = random.next64() >> 11;
	return n != 0ULL ? as_double_cast(n) : 1.0;
}

template <typename TRandom>
float rand_float_oo_cast(TRandom& random)
{
	uint32_t n;
	do
	{
		n = random.next32();
	} while (n <= 0x000000FFU); // If true, then the highest 24 bits must all be zero.
	return as_float_cast(n >> 8);
}

template <typename TRandom>
double rand_double_oo_cast(TRandom& random)
{
	uint64_t n;
	do
	{
		n = random.next64();
	} while (n <= 0x00000000000007FFULL); // If true, then the highest 53 bits must all be zero.
	return as_double_cast(n >> 11);
}

template <typename TRandom>
float rand_float_cc_cast(TRandom& random)
{
	uint32_t n = random.next32();
	// First check if the upper 8 bits are all set.
	// If not, short circuit to the else block.
	// If so, carry on with the messy check.
	if (n >= 0xFF000000U && rand_probability(random, 0x00000100U, 0x01000001U))
	{
		return 0.0f;
	}
	else
	{
		return as_float_cast((n & 0x00FFFFFFU) + 1U);
	}
}

template <typename TRandom>
double rand_double_cc_cast(TRandom& random)
{
	uint64_t n = random.next64();
	// First check if the upper 11 bits are all set.
	// If not, short circuit to the else block.
	// If so, carry on with the messy check.
	if (n >= 0xFFE0000000000000ULL && rand_probability(random, 0x00000800ULL, 0x0020000000000001ULL))
	{
		return 0.0;
	}
	else
	{
		return as_double_cast((n & 0x001FFFFFFFFFFFFFULL) + 1ULL);
	}
}

template <typename TRandom>
float rand_float_co_div(TRandom& random)
{
	return static_cast<float>(random.next32()) / 4294967808.0f;
}

template <typename TRandom>
double rand_double_co_div(TRandom& random)
{
	return static_cast<double>(random.next64()) / 18446744073709555712.0;
}

template <typename TRandom>
float rand_float_oc_div(TRandom& random)
{
	return (static_cast<float>(random.next32()) + 1.0f) / 4294967296.0f;
}

template <typename TRandom>
double rand_double_oc_div(TRandom& random)
{
	return (static_cast<double>(random.next64()) + 1.0) / 18446744073709551616.0;
}

template <typename TRandom>
float rand_float_oo_div(TRandom& random)
{
	return (static_cast<float>(random.next32()) + 1.0f) / 4294967808.0f;
}

template <typename TRandom>
double rand_double_oo_div(TRandom& random)
{
	return (static_cast<double>(random.next64()) + 1.0) / 18446744073709555712.0;
}

template <typename TRandom>
float rand_float_cc_div(TRandom& random)
{
	return static_cast<float>(random.next32()) / 4294967296.0f;
}

template <typename TRandom>
double rand_double_cc_div(TRandom& random)
{
	return static_cast<double>(random.next64()) / 18446744073709551616.0;
}

template <typename TFunc, typename TRandom>
std::string validate_float_distribution(TFunc rand_func, TRandom& random, int loop_count, int iterations_per_loop, bool inclusive_lower, bool inclusive_upper, bool cast)
{
	std::map<float, int> f_count;
	std::vector<float> f_invalid;
	float superZero;
	float half;
	float subOne;

	float min_value;
	float max_value;
	int min_count;
	int max_count;
	double average_count;
	double deviance_sum;

	if (cast)
	{
		if (inclusive_lower)
		{
			f_count.insert(std::make_pair(0.0f, 0));
		}
		for (uint32_t i = 1U; i < 0x01000000U; ++i)
		{
			f_count.insert(std::make_pair(as_float_cast(i), 0));
		}
		if (inclusive_upper)
		{
			f_count.insert(std::make_pair(1.0f, 0));
		}

		superZero = as_float_cast(0x00000001U);
		half = as_float_cast(0x00800000U);
		subOne = as_float_cast(0x00FFFFFFU);
	}
	else
	{
		for (uint32_t i = inclusive_lower ? 0U : 1U; i < 0x00800000U; ++i)
		{
			f_count.insert(std::make_pair(as_float(0x3F800000U | i) - 1.0f, 0));
		}
		if (inclusive_upper)
		{
			f_count.insert(std::make_pair(1.0f, 0));
		}

		superZero = as_float(0x3F800001U) - 1.0f;
		half = 0.5f;
		subOne = as_float(0x3FFFFFFFU) - 1.0f;
	}

	double unique_values = (double)f_count.size();

	for (int loop = 0; loop < loop_count; ++loop)
	{
		std::cout << "Loop " << loop << std::endl;
		for (int i = 0; i < iterations_per_loop; ++i)
		{
			float f = rand_func(random);
			auto h = f_count.find(f);
			if (h != f_count.end())
			{
				h->second = h->second + 1;
			}
			else
			{
				f_invalid.push_back(f);
			}
		}
	}

	min_value = f_count.begin()->first;
	max_value = min_value;
	min_count = f_count.begin()->second;
	max_count = min_count;
	average_count = (double)loop_count * (double)iterations_per_loop / unique_values;
	deviance_sum = 0.0;
	for (const auto count : f_count)
	{
		if (count.second < min_count)
		{
			min_value = count.first;
			min_count = count.second;
		}
		if (count.second > max_count)
		{
			max_value = count.first;
			max_count = count.second;
		}
		double delta = (double)count.second - average_count;
		deviance_sum += delta * delta;
	}

	std::stringstream ss;

	ss << "Validating rand_float_" << (inclusive_lower ? 'c' : 'o') << (inclusive_upper ? 'c' : 'o') << (cast ? "" : "_cast") << "()" << std::endl;
	ss << "Unique Values = " << f_count.size() << std::endl;
	for (auto f : f_invalid)
	{
		ss << "Invalid number generated:  " << std::setprecision(8) << std::fixed << f << std::endl;
	}
	ss << "Average Count = " << std::setprecision(16) << std::fixed << (double)loop_count * (double)iterations_per_loop / unique_values << std::endl;
	ss << "Std Deviation = " << std::setprecision(16) << std::fixed << std::sqrt(deviance_sum / (double)loop_count / (double)iterations_per_loop) << std::endl;
	ss << "Minimum Count = " << min_count << " (" << std::setprecision(8) << std::fixed << min_value << ")" << std::endl;
	ss << "Maximum Count = " << max_count << " (" << std::setprecision(8) << std::fixed << max_value << ")" << std::endl;
	if (inclusive_lower) ss << "Zero Count    = " << f_count[0.0f] << std::endl;
	ss << "Sup-Zro Count = " << f_count[superZero] << std::endl;
	ss << "Half Count    = " << f_count[half] << std::endl;
	ss << "Sub-One Count = " << f_count[subOne] << std::endl;
	if (inclusive_upper) ss << "One Count     = " << f_count[1.0f] << std::endl;
	ss << std::endl;
	return ss.str();
}

template <typename TSource>
struct sink_type_traits
{
	typedef TSource sink_type;
	static sink_type sink(TSource source) { return source; }
};

template <>
struct sink_type_traits<float>
{
	typedef uint32_t sink_type;
	static sink_type sink(float source) { return reinterpret_cast<sink_type&&>(source); }
};

template <>
struct sink_type_traits<double>
{
	typedef uint64_t sink_type;
	static sink_type sink(double source) { return reinterpret_cast<sink_type&&>(source); }
};

template <typename TFunc>
auto rand_loop(TFunc rand_func, uint64_t iterations) -> typename sink_type_traits<decltype(rand_func())>::sink_type
{
	typename sink_type_traits<decltype(rand_func())>::sink_type sink;
	uint64_t outer_loops = iterations >> 4;
	for (uint64_t i = 0ULL; i < outer_loops; ++i)
	{
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());

		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());

		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());

		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
		sink ^= sink_type_traits<decltype(rand_func())>::sink(rand_func());
	}
	return sink;
}

template <typename TFunc>
double measure_operations(TFunc rand_func, double warmup_time, double measure_time, std::string op_name, double baseline_ops_per_second = 0.0)
{
	uint64_t warmup_ticks = get_tick_count(warmup_time);
	uint64_t measure_ticks = get_tick_count(measure_time);
	uint64_t target_warmup_batch_ticks = std::min(get_tick_count(1.0), warmup_ticks / 4ULL);
	uint64_t target_measure_batch_ticks = std::min(get_tick_count(1.0), warmup_ticks / 16ULL);

	uint64_t batch_iteration_count = 1024ULL;
	uint64_t warmup_iteration_count = 0ULL;
	uint64_t measure_iteration_count = 0ULL;

	uint64_t current_tick;
	uint64_t elapsed_ticks;

	typename sink_type_traits<decltype(rand_func())>::sink_type sink = 0;

	uint64_t warmup_start_tick = get_current_tick();

	while (true)
	{
		sink ^= rand_loop(rand_func, batch_iteration_count);

		warmup_iteration_count += batch_iteration_count;

		current_tick = get_current_tick();
		elapsed_ticks = current_tick - warmup_start_tick;

		if (elapsed_ticks >= warmup_ticks)
		{
			break;
		}

		batch_iteration_count = batch_iteration_count << 1;
		uint64_t expected_batch_ticks = batch_iteration_count * elapsed_ticks / warmup_iteration_count;
		uint64_t next_batch_ticks = std::min(expected_batch_ticks, target_warmup_batch_ticks);
		next_batch_ticks = std::min(next_batch_ticks, warmup_ticks - elapsed_ticks);
		batch_iteration_count = next_batch_ticks * warmup_iteration_count / elapsed_ticks;
		batch_iteration_count = (batch_iteration_count + 15ULL) & ~0xFULL;
	}

	batch_iteration_count = target_measure_batch_ticks * warmup_iteration_count / elapsed_ticks;
	batch_iteration_count = (batch_iteration_count + 15ULL) & ~0xFULL;

	sink = 0;

	uint64_t measure_start_tick = get_current_tick();

	while (true)
	{
		sink ^= rand_loop(rand_func, batch_iteration_count);

		measure_iteration_count += batch_iteration_count;

		current_tick = get_current_tick();
		elapsed_ticks = current_tick - measure_start_tick;

		if (elapsed_ticks >= measure_ticks)
		{
			break;
		}

		batch_iteration_count = batch_iteration_count << 1;
		uint64_t expected_batch_ticks = batch_iteration_count * elapsed_ticks / measure_iteration_count;
		uint64_t next_batch_ticks = std::min(expected_batch_ticks, target_measure_batch_ticks);
		next_batch_ticks = std::min(next_batch_ticks, measure_ticks - elapsed_ticks);
		batch_iteration_count = next_batch_ticks * measure_iteration_count / elapsed_ticks;
		batch_iteration_count = (batch_iteration_count + 15ULL) & ~0xFULL;
	}

	double ops_per_second = static_cast<double>(measure_iteration_count) * static_cast<double>(get_tick_count(1.0)) / static_cast<double>(elapsed_ticks);
	double nanoseconds_per_op = 1000000000.0 / ops_per_second;
	double baseline_nanoseconds_per_op = 1000000000.0 / baseline_ops_per_second;

	std::cout << std::setw(20) << op_name << ",  " << std::setprecision(0) << std::fixed << std::setw(9) << ops_per_second << " ops/s"
		<< ",    " << std::setprecision(5) << std::fixed << std::setw(9) << nanoseconds_per_op << " ns/op";

	if (baseline_ops_per_second != 0.0)
	{
		double overhead = baseline_ops_per_second / ops_per_second - 1.0;
		std::cout << ",    " << std::setprecision(5) << std::fixed << std::setw(9) << (nanoseconds_per_op - baseline_nanoseconds_per_op) << " ns/op"
			<< ",    " << std::setprecision(3) << std::fixed << std::setw(7) << overhead * 100.0 << "%";
	}

	std::cout << ",    (sink = " << sink << ")";

	std::cout << std::endl;

	return ops_per_second;
}

//int main()
//{
//	// Use these functions to statistically validate distributions.
//	std::vector<std::future<std::string>> validation_results;
//	// To avoid false sharing with threads, spread out some random engines in memory.
//	std::vector<xorshift_128_plus> random_engines;
//	random_engines.resize(64);
//	random_engines[0] = xorshift_128_plus(0xA6E9377DAF75BDFEULL, 0x863F5CB508510D95ULL);
//	random_engines[8] = xorshift_128_plus(0xA6E9377DAF75BDFEULL, ~0x863F5CB508510D95ULL);
//	random_engines[16] = xorshift_128_plus(~0xA6E9377DAF75BDFEULL, 0x863F5CB508510D95ULL);
//	random_engines[24] = xorshift_128_plus(~0xA6E9377DAF75BDFEULL, ~0x863F5CB508510D95ULL);
//	random_engines[32] = xorshift_128_plus(0xA6E9377DAF75BDFEULL, 0x863F5CB508510D95ULL);
//	random_engines[40] = xorshift_128_plus(0xA6E9377DAF75BDFEULL, ~0x863F5CB508510D95ULL);
//	random_engines[48] = xorshift_128_plus(~0xA6E9377DAF75BDFEULL, 0x863F5CB508510D95ULL);
//	random_engines[56] = xorshift_128_plus(~0xA6E9377DAF75BDFEULL, ~0x863F5CB508510D95ULL);
//	//validation_results.push_back(std::async(std::launch::async, [&random_engines]() -> std::string { return validate_float_distribution(rand_float_co<xorshift_128_plus>, random_engines[0], 10, 100000000, true, false, false); }));
//	//validation_results.push_back(std::async(std::launch::async, [&random_engines]() -> std::string { return validate_float_distribution(rand_float_oc<xorshift_128_plus>, random_engines[8], 10, 100000000, false, true, false); }));
//	//validation_results.push_back(std::async(std::launch::async, [&random_engines]() -> std::string { return validate_float_distribution(rand_float_oo<xorshift_128_plus>, random_engines[16], 10, 100000000, false, false, false); }));
//	//validation_results.push_back(std::async(std::launch::async, [&random_engines]() -> std::string { return validate_float_distribution(rand_float_cc<xorshift_128_plus>, random_engines[24], 10, 100000000, true, true, false); }));
//	//validation_results.push_back(std::async(std::launch::async, [&random_engines]() -> std::string { return validate_float_distribution(rand_float_co_cast<xorshift_128_plus>, random_engines[32], 10, 100000000, true, false, true); }));
//	//validation_results.push_back(std::async(std::launch::async, [&random_engines]() -> std::string { return validate_float_distribution(rand_float_oc_cast<xorshift_128_plus>, random_engines[40], 10, 100000000, false, true, true); }));
//	//validation_results.push_back(std::async(std::launch::async, [&random_engines]() -> std::string { return validate_float_distribution(rand_float_oo_cast<xorshift_128_plus>, random_engines[48], 10, 100000000, false, false, true); }));
//	//validation_results.push_back(std::async(std::launch::async, [&random_engines]() -> std::string { return validate_float_distribution(rand_float_cc_cast<xorshift_128_plus>, random_engines[56], 10, 100000000, true, true, true); }));
//
//	bool ready;
//	do
//	{
//		ready = true;
//		for (auto&& validation_result : validation_results)
//		{
//			if (validation_result.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready)
//			{
//				ready = false;
//				break;
//			}
//		}
//	} while (!ready);
//
//	for (auto&& validation_result : validation_results)
//	{
//		std::cout << validation_result.get();
//	}
//
//	// Use these functions to measure performance.
//	std::cout << "Press any key to begin." << std::endl;
//	std::cin.get();
//
//	prepare_thread();
//
//	double baseline;
//
//	auto random_engine = xorshift_128_plus(0xA6E9377DAF75BDFEULL, 0x863F5CB508510D95ULL);
//	std::ranlux24_base random_engine_32;
//	std::ranlux48_base random_engine_64;
//	std::uniform_real<float> flt_co;
//	std::uniform_real<double> dbl_co;
//
//	double warmup_duration = 1.0;
//	double measure_duration = 60.0;
//
//	baseline = measure_operations([&random_engine]() -> uint32_t { return random_engine.next32(); }, warmup_duration, measure_duration, "rand_next32");
//	std::cout << std::endl;
//
//	measure_operations([&random_engine]() -> float { return rand_float_co(random_engine); }, warmup_duration, measure_duration, "rand_float_co", baseline);
//	measure_operations([&random_engine]() -> float { return rand_float_oc(random_engine); }, warmup_duration, measure_duration, "rand_float_oc", baseline);
//	measure_operations([&random_engine]() -> float { return rand_float_oo(random_engine); }, warmup_duration, measure_duration, "rand_float_oo", baseline);
//	measure_operations([&random_engine]() -> float { return rand_float_cc(random_engine); }, warmup_duration, measure_duration, "rand_float_cc", baseline);
//	std::cout << std::endl;
//
//	measure_operations([&random_engine]() -> float { return rand_float_co_cast(random_engine); }, warmup_duration, measure_duration, "rand_float_co_cast", baseline);
//	measure_operations([&random_engine]() -> float { return rand_float_oc_cast(random_engine); }, warmup_duration, measure_duration, "rand_float_oc_cast", baseline);
//	measure_operations([&random_engine]() -> float { return rand_float_oo_cast(random_engine); }, warmup_duration, measure_duration, "rand_float_oo_cast", baseline);
//	measure_operations([&random_engine]() -> float { return rand_float_cc_cast(random_engine); }, warmup_duration, measure_duration, "rand_float_cc_cast", baseline);
//	std::cout << std::endl;
//
//	measure_operations([&random_engine]() -> float { return rand_float_co_div(random_engine); }, warmup_duration, measure_duration, "rand_float_co_div", baseline);
//	measure_operations([&random_engine]() -> float { return rand_float_oc_div(random_engine); }, warmup_duration, measure_duration, "rand_float_oc_div", baseline);
//	measure_operations([&random_engine]() -> float { return rand_float_oo_div(random_engine); }, warmup_duration, measure_duration, "rand_float_oo_div", baseline);
//	measure_operations([&random_engine]() -> float { return rand_float_cc_div(random_engine); }, warmup_duration, measure_duration, "rand_float_cc_div", baseline);
//	std::cout << std::endl;
//
//	baseline = measure_operations([&]() -> uint32_t { return random_engine_32(); }, warmup_duration, measure_duration, "ranlux24");
//	measure_operations([&random_engine_32, flt_co]() -> float { return flt_co(random_engine_32); }, warmup_duration, measure_duration, "flt_co(ranlux24)", baseline);
//	std::cout << std::endl;
//
//	std::cout << std::endl;
//	std::cout << std::endl;
//
//	baseline = measure_operations([&random_engine]() -> uint64_t { return random_engine.next64(); }, warmup_duration, measure_duration, "rand_next64");
//	std::cout << std::endl;
//
//	measure_operations([&random_engine]() -> double { return rand_double_co(random_engine); }, warmup_duration, measure_duration, "rand_double_co", baseline);
//	measure_operations([&random_engine]() -> double { return rand_double_oc(random_engine); }, warmup_duration, measure_duration, "rand_double_oc", baseline);
//	measure_operations([&random_engine]() -> double { return rand_double_oo(random_engine); }, warmup_duration, measure_duration, "rand_double_oo", baseline);
//	measure_operations([&random_engine]() -> double { return rand_double_cc(random_engine); }, warmup_duration, measure_duration, "rand_double_cc", baseline);
//	std::cout << std::endl;
//
//	measure_operations([&random_engine]() -> double { return rand_double_co_cast(random_engine); }, warmup_duration, measure_duration, "rand_double_co_cast", baseline);
//	measure_operations([&random_engine]() -> double { return rand_double_oc_cast(random_engine); }, warmup_duration, measure_duration, "rand_double_oc_cast", baseline);
//	measure_operations([&random_engine]() -> double { return rand_double_oo_cast(random_engine); }, warmup_duration, measure_duration, "rand_double_oo_cast", baseline);
//	measure_operations([&random_engine]() -> double { return rand_double_cc_cast(random_engine); }, warmup_duration, measure_duration, "rand_double_cc_cast", baseline);
//	std::cout << std::endl;
//
//	measure_operations([&random_engine]() -> double { return rand_double_co_div(random_engine); }, warmup_duration, measure_duration, "rand_double_co_div", baseline);
//	measure_operations([&random_engine]() -> double { return rand_double_oc_div(random_engine); }, warmup_duration, measure_duration, "rand_double_oc_div", baseline);
//	measure_operations([&random_engine]() -> double { return rand_double_oo_div(random_engine); }, warmup_duration, measure_duration, "rand_double_oo_div", baseline);
//	measure_operations([&random_engine]() -> double { return rand_double_cc_div(random_engine); }, warmup_duration, measure_duration, "rand_double_cc_div", baseline);
//	std::cout << std::endl;
//
//	baseline = measure_operations([&]() -> uint64_t { return random_engine_64(); }, warmup_duration, measure_duration, "ranlux48");
//	measure_operations([&random_engine_64, dbl_co]() -> double { return dbl_co(random_engine_64); }, warmup_duration, measure_duration, "dbl_co(ranlux48)", baseline);
//	std::cout << std::endl;
//
//	std::cout << std::endl;
//	std::cout << std::endl;
//
//	std::cout << "Measurements complete." << std::endl;
//	std::cin.get();
//
//	return 0;
//}