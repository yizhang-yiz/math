#include <stan/math/prim/scal/err/is_bounded.hpp>
#include <gtest/gtest.h>

using stan::math::is_bounded;

TEST(ErrorHandlingScalar, is_bounded_x) {
  double x = 0;
  double low = -1;
  double high = 1;
 
  EXPECT_TRUE(is_bounded(x, low, high))
    << "is_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  x = low;
  EXPECT_TRUE(is_bounded(x, low, high)) 
    << "is_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = high;
  EXPECT_TRUE(is_bounded(x, low, high)) 
    << "is_bounded should be TRUE with x: " << x << " equal to the lower bound: " << low;

  x = low-1;
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;  
  
  x = high+1;
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;  
}

TEST(ErrorHandlingScalar, is_bounded_Low) {
  double x = 0;
  double low = -1;
  double high = 1;
 
  EXPECT_TRUE(is_bounded(x, low, high))
    << "is_bounded should be true x: " << x << " and bounds: " << low << ", " << high;
  
  low = -std::numeric_limits<double>::infinity();
  EXPECT_TRUE(is_bounded(x, low, high))
    << "is_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}
TEST(ErrorHandlingScalar, is_bounded_High) {
  double x = 0;
  double low = -1;
  double high = 1;
 
  EXPECT_TRUE(is_bounded(x, low, high))
    << "is_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  high = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(is_bounded(x, low, high)) 
    << "is_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  high = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  high = -std::numeric_limits<double>::infinity();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
}
TEST(ErrorHandlingScalar,CheckBounded_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  double x = 0;
  double low = -1;
  double high = 1;

  EXPECT_FALSE(is_bounded(nan, low, high));
  EXPECT_FALSE(is_bounded(x, nan, high));
  EXPECT_FALSE(is_bounded(x, low, nan));
  EXPECT_FALSE(is_bounded(nan, nan, high));
  EXPECT_FALSE(is_bounded(nan, low, nan));
  EXPECT_FALSE(is_bounded(x, nan, nan));
  EXPECT_FALSE(is_bounded(nan, nan, nan));
}
