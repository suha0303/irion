#include "powertrain.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
int main(int argc, char **argv) {
  try {
    x500::Powertrain core;
    double duration = argc > 1 ? std::stod(argv[1]) : 10;
    double soc = argc > 2 ? std::stod(argv[2]) : .9;
    double duty = argc > 3 ? std::stod(argv[3]) : .55;
    double capacity = argc > 4 ? std::stod(argv[4]) : 6;
    if (duration <= 0 || duration > 3600 || soc <= 0 || soc > 1)
      throw std::invalid_argument("bench bounds");
    core.p.inertia *= argc > 5 ? std::stod(argv[5]) : 1;
    core.state.soc = soc;
    core.p.capacity_ah = capacity;
    std::cout << std::setprecision(12)
              << "time_s,voltage_v,current_a,omega_rad_s,thrust_total_n,soc,"
                 "charge_c,energy_residual_j,bus_energy_j\n";
    for (int k = 0; k < duration * 1000; ++k) {
      auto o = core.step(.001, {duty, duty, duty, duty}, true);
      if (k % 100 == 99)
        std::cout << (k + 1) * .001 << ',' << o.voltage << ',' << o.current
                  << ',' << core.state.omega[0] << ',' << 4 * o.thrust[0] << ','
                  << core.state.soc << ',' << core.state.charge_c << ','
                  << core.energy_residual() << ',' << core.state.bus_energy
                  << '\n';
      if (o.status == x500::Output::depleted)
        break;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
