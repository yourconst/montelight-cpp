#pragma once

#include "gen_rand.hpp"

#include <stdlib.h>

double simple_static_random()
{
	static thread_local auto engine = xorshift_128_plus(0xA6E9377DAF75BDFEULL, 0x863F5CB508510D95ULL);
	uint64_t n;
	do
	{
		n = engine.next64();
	} while (n <= 0x0000000000000FFFULL); // If true, then the highest 52 bits must all be zero.
	return as_double(0x3FF0000000000000ULL | (n >> 12)) - 1.0;
}

#define drand() simple_static_random()

//thread_local auto simple_random_engine = xorshift_128_plus(0xA6E9377DAF75BDFEULL, 0x863F5CB508510D95ULL);
//
//double simple_random()
//{
//	uint64_t n;
//	do
//	{
//		n = simple_random_engine.next64();
//	} while (n <= 0x0000000000000FFFULL); // If true, then the highest 52 bits must all be zero.
//	return as_double(0x3FF0000000000000ULL | (n >> 12)) - 1.0;
//}



//const double INV_RAND_MAX = 1.0 / (RAND_MAX + 1);
//
//#ifdef _WIN32
//    #define drand() INV_RAND_MAX * rand()
//    //#define drand() (double) ((0x3FF0000000000000ULL) | (rand() << 32))
//#else
//    #define drand() drand48()
//#endif

//float drand() {
//    #ifdef _WIN32
//      return double(rand()) / RAND_MAX;
//    #else
//      return drand48();
//    #endif
//}

//#define drand() ((double) (0x3FF0000000000000ULL | (unsigned long long) (rand())))

//union D_ULL {
//    unsigned long long ull;
//    double d;
//};
//
//void putDRand(void * d) {
//    ((D_ULL * )d)->ull = (0x3FF0000000000000ULL | (((unsigned long long) rand()) << 37));
//    ((D_ULL*)d)->d -= 1;
//}
//#define PUT_D_RAND(v) ((D_ULL * ) &v)->ull = (0x3FF0000000000000ULL | (((unsigned long long) rand()) << 37)); ((D_ULL*) &v)->d -= 1;
//double drand(D_ULL ) {
//    D_ULL s = {
//        (0x3FF0000000000000ULL | (((unsigned long long) rand()) << 37))
//        /*&
//         0xFFEFFFFFFFFFFFFFULL*/
//    };
//    return s.d - 1;
//}

//void testDrand() {
//    /*double a = .0;
//    double b = 1.0;
//    double c = drand();
//    unsigned long long d = rand();
//    auto e = 0x3FF0000000000000ULL;
//    auto r = e | (((unsigned long long) rand()) << 36);
//    auto rr = (double) (e | (((unsigned long long) rand()) << 36));
//
//    binary(& a);
//    binary(& b);
//    binary(& c);
//    cout << c << endl;
//    binary(& d);
//    cout << d << endl;
//    binary(& e);
//    binary(& r);
//    binary(& rr);
//
//    double sum = 0;
//    unsigned cnt = 100;
//    for (auto i = 0; i < cnt; ++i) {
//        auto d = drand16();
//        sum += d;
//        binary(&d, false);
//        cout << " - " << d << endl;
//        d -= 1;
//        cout << "   " << d << endl;
//    }
//    r = e | (((unsigned long long) rand()) << 36);
//    binary(&r);
//
//    cout << (sum / cnt) << endl;
//
//    return 0;*/
//}





//thread_local auto random_engine = xorshift_128_plus(0xA6E9377DAF75BDFEULL, 0x863F5CB508510D95ULL);
//
//double myRand() { return rand_double_oo(random_engine); }
//
//#define MY_RAND() rand_double_oo(random_engine)

//void main() {
//    auto random_engine2 = xorshift_128_plus(0xA6E9377DAF75BDFEULL, 0x863F5CB508510D95ULL);
//    std::mt19937 gen(time(0));
//    std::uniform_real_distribution<double> urd(0, 1);
//
//    unsigned cnt = 1e9;
//    double sum;
//    unsigned tm;
//
//    sum = 0;
//    tm = getCurrentTime();
//    for (unsigned i = 0; i < cnt; ++i)
//        sum += drand();
//    cout << "Average: " << (sum / cnt) << " , Time: " << ((getCurrentTime() - tm) / 1e4) << " ms" << endl;
//
//    sum = 0;
//    tm = getCurrentTime();
//    for (unsigned i = 0; i < cnt; ++i)
//        sum += urd(gen);
//    cout << "Average: " << (sum / cnt) << " , Time: " << ((getCurrentTime() - tm) / 1e4) << " ms" << endl;
//
//    sum = 0;
//    tm = getCurrentTime();
//    for (unsigned i = 0; i < cnt; ++i)
//        sum += rand_double_oo(random_engine);
//    cout << "Average: " << (sum / cnt) << " , Time: " << ((getCurrentTime() - tm) / 1e4) << " ms" << endl;
//
//    sum = 0;
//    tm = getCurrentTime();
//    for (unsigned i = 0; i < cnt; ++i)
//        sum += rand_double_oo(random_engine2);
//    cout << "Average: " << (sum / cnt) << " , Time: " << ((getCurrentTime() - tm) / 1e4) << " ms" << endl;
//
//    sum = 0;
//    tm = getCurrentTime();
//    for (unsigned i = 0; i < cnt; ++i)
//        sum += simple_random();
//    cout << "Average: " << (sum / cnt) << " , Time: " << ((getCurrentTime() - tm) / 1e4) << " ms" << endl;
//
//    sum = 0;
//    tm = getCurrentTime();
//    for (unsigned i = 0; i < cnt; ++i)
//        sum += simple_static_random();
//    cout << "Average: " << (sum / cnt) << " , Time: " << ((getCurrentTime() - tm) / 1e4) << " ms" << endl;
//
//    sum = 0;
//    tm = getCurrentTime();
//    for (unsigned i = 0; i < cnt; ++i)
//        sum += myRand();
//    cout << "Average: " << (sum / cnt) << " , Time: " << ((getCurrentTime() - tm) / 1e4) << " ms" << endl;
//
//    sum = 0;
//    tm = getCurrentTime();
//    for (unsigned i = 0; i < cnt; ++i)
//        sum += MY_RAND();
//    cout << "Average: " << (sum / cnt) << " , Time: " << ((getCurrentTime() - tm) / 1e4) << " ms" << endl;
//}