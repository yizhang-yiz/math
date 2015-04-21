#include <stan/math/prim/scal/fun/prob_free.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, prob_free) {
  double L = 0.0;
  double U = 1.0;
  double y = 0.4;
  EXPECT_FLOAT_EQ(stan::math::logit((y - L) / (U - L)),
                  stan::math::prob_free(y));
}
TEST(MathPrim, prob_free_exception) {
  EXPECT_THROW (stan::math::prob_free(1.1), std::domain_error);
  EXPECT_THROW (stan::math::prob_free(-0.1), std::domain_error);
}
