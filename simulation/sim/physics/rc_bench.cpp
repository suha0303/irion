#include "powertrain.hpp"
#include <iomanip>
#include <iostream>
int main() {
  x500::Powertrain a;
  std::cout << std::setprecision(14)
            << "t,current,soc,vp,temperature,voltage,energy_residual\n";
  for (int k = 0; k < 120000; ++k) {
    a.p.auxiliary_a = k < 30000 ? 10 : (k < 60000 ? 20 : 0);
    auto o = a.step(.001, {0, 0, 0, 0}, false);
    if (k % 100 == 99)
      std::cout << (k + 1) * .001 << ',' << o.current << ',' << a.state.soc
                << ',' << a.state.polarization << ',' << a.state.temperature
                << ',' << o.voltage << ',' << a.battery_energy_residual()
                << '\n';
  }
}
