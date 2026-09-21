#include "powertrain.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
  try {
    if (argc != 3) throw std::runtime_error("battery and propulsion files required");
    x500::Parameters p;
    std::ifstream b(argv[1]), m(argv[2]);
    b >> p.r0 >> p.r1 >> p.c1 >> p.capacity_ah;
    for (auto &v : p.ocv_cell) b >> v;
    m >> p.kt_prop >> p.kq_prop >> p.friction >> p.coulomb >> p.esc_efficiency >> p.resistance;
    if (!b || !m) throw std::runtime_error("runtime files");
    p.calibrated = true;
    p.battery_heat_capacity = x500::profile::battery_heat_capacity;
    p.battery_cooling = x500::profile::battery_cooling;
    x500::Powertrain pulse(p);
    auto initial_soc = pulse.state.soc;
    for (int k=0; k<2000; ++k) pulse.step(.001, {.6,.6,.6,.6}, true);
    double charge_error = std::abs((initial_soc-pulse.state.soc)*3600*p.capacity_ah-pulse.state.charge_c);
    double energy_error = std::abs(pulse.energy_residual());
    double battery_energy_error = std::abs(pulse.battery_energy_residual());
    if (charge_error>1e-6 || energy_error>1e-3 || battery_energy_error>1e-3)
      throw std::runtime_error("charge/energy conservation");
    x500::Powertrain fine(p);
    fine.p.max_substep = .000125;
    for(int k=0;k<4000;++k) fine.step(.0005,{.6,.6,.6,.6},true);
    double convergence = std::abs(fine.state.soc-pulse.state.soc);
    if(convergence>1e-7) throw std::runtime_error("time convergence");
    std::cout<<std::setprecision(16)<<"{\"kind\":\"pulse\",\"seconds\":2,\"charge_error_c\":"<<charge_error
             <<",\"energy_error_j\":"<<energy_error<<",\"battery_energy_error_j\":"<<battery_energy_error
             <<",\"soc_convergence\":"<<convergence<<"}\n";
    for(double ambient : {283.15,298.15}) for(double payload : {0.,.2})
      for(double soc : {.2,.3,.4,.9}) for(double factor : {1.,1.5}) {
        double mass=2.4571076923076927+payload;
        double omega=std::sqrt(mass*9.80665*factor/(4*p.kt_prop));
        double drag=p.friction*omega+p.coulomb*std::tanh(omega);
        double im=(p.kq_prop*omega*omega+drag)/p.ke;
        // Thermal steady-state motor copper resistance, geometry held fixed.
        double delta=(im*im*p.resistance*(1+.00393*(ambient-298.15))+drag*omega)/
                     (p.motor_cooling-im*im*p.resistance*.00393);
        double motor_t=ambient+delta;
        double rm=p.resistance*(1+.00393*(motor_t-298.15));
        double vm=p.ke*omega+rm*im;
        double watts=4*vm*im/p.esc_efficiency;
        x500::Powertrain core(p);
        core.state.soc=soc; core.state.omega.fill(omega); core.state.motor_temperature.fill(motor_t);
        double r=p.r0+p.r1, term=core.ocv(soc)-r*p.auxiliary_a;
        double disc=term*term-4*r*watts;
        if(disc<=0) throw std::runtime_error("no steady solution");
        double voltage=(term+std::sqrt(disc))/2, current=watts/voltage+p.auxiliary_a;
        core.state.polarization=p.r1*current;
        double duty=vm/voltage;
        auto o=core.solve(core.state,{duty,duty,duty,duty},true);
        bool limits=duty<=1 && im<=p.motor_limit && watts/(4*voltage)<=p.dc_limit;
        if(limits && (std::abs(o.voltage-voltage)>1e-6 || std::abs(o.acceleration[0])>1e-3))
          throw std::runtime_error("independent steady equation vs core");
        std::cout<<"{\"kind\":\"steady\",\"ambient_k\":"<<ambient<<",\"payload_kg\":"<<payload
                 <<",\"soc\":"<<soc<<",\"thrust_factor\":"<<factor<<",\"voltage_v\":"<<voltage
                 <<",\"current_a\":"<<current<<",\"duty\":"<<duty<<",\"motor_temperature_k\":"<<motor_t
                 <<",\"motor_current_a\":"<<im<<",\"esc_current_a\":"<<watts/(4*voltage)
                 <<",\"within_actuator_limits\":"<<(limits?"true":"false")
                 <<",\"above_voltage_floor\":"<<(voltage>=12.4?"true":"false")<<"}\n";
      }
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 4;}
}
