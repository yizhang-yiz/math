#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/value_type.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/arr/meta/VectorView.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>

TEST(AgradRevErrorHandlingScalar,is_bounded_X) {
  using stan::math::var;
  using stan::math::is_bounded;
 
  var x = 0;
  var low = -1;
  var high = 1;
 
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

  x = std::numeric_limits<var>::quiet_NaN();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = -std::numeric_limits<var>::infinity();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;

  x = std::numeric_limits<var>::infinity();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << high << ", " << low;
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar,is_bounded_Low) {
  using stan::math::var;
  using stan::math::is_bounded;

  var x = 0;
  var low = -1;
  var high = 1;
 
  EXPECT_TRUE(is_bounded(x, low, high)) 
    << "is_bounded should be true x: " << x << " and bounds: " << low << ", " << high;
  
  low = -std::numeric_limits<var>::infinity();
  EXPECT_TRUE(is_bounded(x, low, high)) 
    << "is_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::quiet_NaN();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  low = std::numeric_limits<var>::infinity();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar,is_bounded_High) {
  using stan::math::var;
  using stan::math::is_bounded;

  var x = 0;
  var low = -1;
  var high = 1;
 
  EXPECT_TRUE(is_bounded(x, low, high)) 
    << "is_bounded should be true x: " << x << " and bounds: " << low << ", " << high;

  high = std::numeric_limits<var>::infinity();
  EXPECT_TRUE(is_bounded(x, low, high)) 
    << "is_bounded should be TRUE with x: " << x << " and bounds: " << low << ", " << high;
  
  high = std::numeric_limits<var>::quiet_NaN();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;

  high = -std::numeric_limits<var>::infinity();
  EXPECT_FALSE(is_bounded(x, low, high))
    << "is_bounded should throw with x: " << x << " and bounds: " << low << ", " << high;
  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, is_bounded_VarCheckUnivariate) {
  using stan::math::var;
  using stan::math::is_bounded;

  var a(5.0);

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(1U,stack_size);
  EXPECT_TRUE(is_bounded(a,4.0,6.0));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(1U,stack_size_after_call);

  stan::math::recover_memory();
}

TEST(AgradRevErrorHandlingScalar, is_bounded_VarCheckVectorized) {
  using stan::math::var;
  using std::vector;
  using stan::math::is_bounded;

  int N = 5;
  vector<var> a;

  for (int i = 0; i < N; ++i)
   a.push_back(var(i));

  size_t stack_size = stan::math::ChainableStack::var_stack_.size();

  EXPECT_EQ(5U,stack_size);
  EXPECT_TRUE(is_bounded(a,-1.0,6.0));

  size_t stack_size_after_call = stan::math::ChainableStack::var_stack_.size();
  EXPECT_EQ(5U,stack_size_after_call);
  stan::math::recover_memory();
}

