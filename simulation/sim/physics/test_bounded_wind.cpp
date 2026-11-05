#include "bounded_wind.hpp"
#include <cassert>
#include <iostream>
#include <limits>

double norm(const std::array<double, 3> &v) {
  return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}
void inside(const std::array<double, 3> &v, const std::string &environment) {
  double speed = norm(v);
  assert(std::isfinite(speed));
  if (environment == "calm") assert(speed == 0);
  else if (environment == "low") assert(speed > 0 && speed <= 2.5);
  else assert(speed > 2.5 && speed <= 5);
}
int main() {
  for (const std::string environment : {"calm", "low", "strong"}) {
    for (double alpha : {.5, 1., 1.5}) for (double angle : {0., 90., 180., 270.}) {
      x500::BoundedWind a(94108, environment, angle, alpha), b = a;
      for (double x : {-1e300, -20., -1., 0., 1., 20., 1e300})
        for (double y : {-1e300, -20., -1., 0., 1., 20., 1e300})
          for (double z : {-1e300, -20., -1., 0., 1., 20., 1e300})
            inside(a.transform({x,y,z}), environment);
      for (int tick = 0; tick <= 480000; ++tick) {
        double time = tick / 1000.;
        auto v = a.sample(time);
        inside(v, environment);
        if (tick % 137 == 0) assert(v == b.sample(time));
      }
      x500::BoundedWind c(94108, environment, angle, alpha);
      for (int knot = 1; knot <= 100; ++knot) {
        auto left = c.sample(knot-1e-4), center = c.sample(knot), right = c.sample(knot+1e-4);
        for (int axis = 0; axis < 3; ++axis) {
          assert(std::abs(left[axis]-center[axis]) < 1e-7);
          assert(std::abs(right[axis]-center[axis]) < 1e-7);
        }
      }
    }
  }
  x500::BoundedWind north(1,"strong",0), east(1,"strong",90);
  assert(north.transform({0,0,0}) == (std::array<double,3>{0,3.75,0}));
  auto e=east.transform({0,0,0});
  assert(std::abs(e[0]-3.75)<1e-14 && std::abs(e[1])<1e-14);
  bool rejected=false;
  try {north.transform({std::numeric_limits<double>::quiet_NaN(),0,0});}
  catch (const std::invalid_argument &) {rejected=true;}
  assert(rejected);
  std::cout << "PASS bounded environments, extreme latent states, all1ms samples, C2 knots, seed and step-size invariance, ENU axes\n";
}
