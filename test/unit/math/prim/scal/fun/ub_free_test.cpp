#include <stan/math/prim/scal/fun/ub_free.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, ub_free) {
  double y = 2.0;
  double U = 4.0;
  EXPECT_FLOAT_EQ(log(-(y - U)), stan::math::ub_free(2.0,4.0));

  EXPECT_FLOAT_EQ(19.765, 
                  stan::math::ub_free(19.765,
                                      std::numeric_limits<double>::infinity()));
}

TEST(MathPrim, ub_free_exception) {
  double ub = 4.0;
  EXPECT_THROW (stan::math::ub_free(ub+0.01, ub), std::domain_error);
}
