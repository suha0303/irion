#include "aerodynamics.hpp"
#include "command.hpp"
#include "powertrain.hpp"
#include <iostream>
#include <numeric>

using namespace x500;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

int main() {
  try {
    Parameters parameters;
    require(rotor_count >= 4 && rotor_count <= 8, "supported rotor count");
    Powertrain core(parameters);
    for (double temperature : core.state.motor_temperature)
      require(temperature == 298.15, "all rotor temperatures initialized");
    Four duty{}, counts{};
    duty.fill(.6);
    counts.fill(600);
    CommandGate gate;
#ifdef CAPSTONE_PROFILE_GENERIC
    require(parameters.series_cells == profile::series_cells,
            "battery series-cell profile");
    require(core.ocv(0) == parameters.series_cells * profile::ocv_cell[0],
            "per-cell OCV and pack series conversion");
    auto accepted = gate.accept(profile::esc_schema, profile::esc_order,
                               "post_mapping_count_0_1000", 1, 1000, 1000,
                               counts);
#else
    auto accepted = gate.accept("x500.esc.request/1", "FR,RL,FL,RR",
                               "post_mapping_count_0_1000", 1, 1000, 1000,
                               counts);
#endif
    require(accepted == "accepted", "profile command accepted");
    require(gate.duty.back() == .6, "last motor command is processed");
    Output output;
    for (int i = 0; i < 2000; ++i)
      output = core.step(.001, duty, true);
    for (double omega : core.state.omega)
      require(omega > 0 && omega == core.state.omega[0],
              "every identical motor is integrated");
    double total_current = parameters.auxiliary_a +
        std::accumulate(output.dc_current.begin(), output.dc_current.end(), 0.);
    require(std::abs(total_current - output.current) < 1e-10,
            "pack current includes every motor");
    require(std::abs(core.energy_residual()) < .002,
            "electromechanical energy conservation");
    require(std::abs(core.battery_energy_residual()) < .002,
            "battery energy conservation");
    require(std::abs((.9 - core.state.soc) * parameters.capacity_ah * 3600 -
                     core.state.charge_c) < 1e-6,
            "charge conservation");
    Output last;
    last.thrust.back() = 5;
    auto force = wrench(parameters, State{}, last, {0, 0, 0}, {0, 0, 0});
    require(force.force[2] == 5, "last rotor contributes to body force");
    const auto &position = profile::positions.back();
    require(std::abs(force.moment[0] - position[1] * 5) < 1e-12 &&
                std::abs(force.moment[1] + position[0] * 5) < 1e-12,
            "profile geometry contributes the last rotor moment");
    std::cout << "PASS profile physics: rotors=" << rotor_count
              << " series_cells=" << parameters.series_cells
              << " voltage=" << output.voltage
              << " current=" << output.current << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 4;
  }
}
