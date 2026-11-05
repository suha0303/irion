#include "joint_phase.hpp"
#include "bounded_wind.hpp"
#include "../adapters/JointPhaseReference.hpp"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

struct Point { uint64_t timestamp{}; float position[3]{}, velocity[3]{}, acceleration[3]{}, yaw{}, yawspeed{}; };
int main() {
  namespace fs = std::filesystem;
  const auto folder = fs::temp_directory_path()/ ("joint-phase-test-"+std::to_string(getpid()));
  fs::create_directory(folder);
  const auto table = folder/"reference.csv";
  for (const int64_t duration : {180000000LL,480000000LL}) {
  setenv("X500_JOINT_DURATION_US",std::to_string(duration).c_str(),1);
  { std::ofstream out(table); for(int64_t i=0;i<=duration/100000;++i) out << i*100000LL << ',' << i/1000. << ",0,-3,0,0,0,0,0,0,0\n"; }
  setenv("X500_JOINT_REFERENCE",table.c_str(),1);
  setenv("X500_JOINT_POLICY_SHA256",std::string(64,'a').c_str(),1);
  for (const std::string name : {"ORIGIN","LANDING","LAND_REQUEST","RECEIPTS"}) {
    const auto path=folder/name;setenv(("X500_JOINT_"+name).c_str(),path.c_str(),1);
  }
  // A missing native readiness must fail, not silently postpone the origin.
  auto child=fork(); assert(child>=0);
  if (!child) {joint_phase::Reference ref; Point p;ref.prepare(59000000,false,0,0,-3,p);std::_Exit(0);}
  int status=0;waitpid(child,&status,0);assert(WIFEXITED(status)&&WEXITSTATUS(status)==74);
  for (int64_t delay : {0LL,84000LL,180000LL}) {
    fs::remove(folder/"ORIGIN");
    joint_phase::Reference ref;Point point;
    ref.prepare(57000000,true,0,0,-3,point);ref.prepare(58000000,true,0,0,-3,point);
    ref.prepare(60000000+delay,true,0,0,-3,point);
    assert(!fs::exists(folder/"ORIGIN")); // prepare/publication is not consumption.
    assert(point.position[0]==0);
    ref.consumed(60000000+delay,true,false,point);
    assert(fs::exists(folder/"ORIGIN"));
    ref.prepare(60100000+delay,true,0,0,-3,point);assert(std::abs(point.position[0]-.001f)<1e-8f);
    ref.consumed(60100000+delay,true,false,point);
    ref.prepare(60000000+duration+delay,true,0,0,-3,point);assert(ref.land_due(60000000+duration+delay));
    ref.consumed(60000000+duration+delay,true,false,point);ref.land_published(60000000+duration+delay);
    assert(!ref.land_due(60000000+duration+delay));
    ref.consumed(60010000+duration+delay,false,true,point);assert(fs::exists(folder/"LANDING"));
  }
  }
  x500::JointWeatherClock clock;clock.notice(60000000,60180000);
  assert(clock.phase_seconds(60000000)==0&&clock.phase_seconds(61000000)==0);
  assert(std::abs(clock.phase_seconds(61001000)-.001)<1e-12);
  bool rejected=false;try {x500::JointWeatherClock bad;bad.notice(60000000,60500001);}catch(const std::invalid_argument&){rejected=true;}assert(rejected);
  for (int64_t delay : {84000LL,180000LL}) {
    x500::JointWeatherClock shifted;shifted.notice(60000000+delay,60000000+delay+1000);
    x500::BoundedWind a(95101,"strong"),b(95101,"strong");
    for(int i=0;i<=480000;++i) {
      auto x=a.sample(clock.phase_seconds(60000000+i*1000LL));
      auto y=b.sample(shifted.phase_seconds(60000000+delay+i*1000LL));
      for(int axis=0;axis<3;++axis)assert(x[axis]==y[axis]);
    }
  }
  // Native kernel starts with zero slope/curvature, joining the initial plateau C2.
  x500::BoundedWind wind(95101,"strong");auto left=wind.sample(0);auto right=wind.sample(1e-6);
  for(int i=0;i<3;++i)assert(std::abs(left[i]-right[i])<1e-12);
  fs::remove_all(folder);
}
