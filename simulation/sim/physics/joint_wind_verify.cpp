#include "bounded_wind.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
int main(int argc,char **argv) { try {
  if(argc!=7&&argc!=8) throw std::invalid_argument("csv origin seed environment direction variation [duration_s]");
  const std::string duration_arg=argc==8?argv[7]:"480";
  if(duration_arg!="180"&&duration_arg!="480")throw std::invalid_argument("duration must be180 or480 seconds");
  const long long duration_us=std::stoll(duration_arg)*1000000;
  const auto origin=std::stoll(argv[2]);
  x500::BoundedWind wind(static_cast<unsigned>(std::stoul(argv[3])),argv[4],std::stod(argv[5]),std::stod(argv[6]));
  std::ifstream input(argv[1]);std::string line;
  if(!std::getline(input,line)||line!="source_us,weather_seconds,east,north,up") throw std::runtime_error("joint wind header");
  long long previous=-1,mission=0,total=0;double maximum=0;
  while(std::getline(input,line)) {
    std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
    long long stamp;double phase,x,y,z;std::string extra;
    if(!(row>>stamp>>phase>>x>>y>>z)||(row>>extra)) throw std::runtime_error("joint wind row");
    if(previous>=0&&stamp-previous!=1000)throw std::runtime_error("missing/duplicate physics tick");
    previous=stamp;++total;
    const double wanted=std::max(0.,double(stamp-origin-1000000)/1e6);
    if(!std::isfinite(phase)||std::abs(phase-wanted)>1e-12)throw std::runtime_error("actual physical wind phase");
    const auto actual=wind.sample(wanted);
    if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(z))throw std::runtime_error("finite wind");
    maximum=std::max({maximum,std::abs(x-actual[0]),std::abs(y-actual[1]),std::abs(z-actual[2])});
    if(maximum>2e-9)throw std::runtime_error("physical wind differs from declared function");
    if(stamp>=origin&&stamp<=origin+duration_us) {
      if(stamp-origin!=mission*1000)throw std::runtime_error("phase tick coverage");
      ++mission;
    }
  }
  if(!input.eof()||mission!=duration_us/1000+1)throw std::runtime_error("full declared mission wind coverage");
  std::cout<<std::setprecision(17)<<"{\"status\":\"passed\",\"full_run_samples\":"<<total<<",\"mission_samples\":"<<mission<<",\"max_abs_component_error_mps\":"<<maximum<<"}\n";
  return 0;
} catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 4;} }
