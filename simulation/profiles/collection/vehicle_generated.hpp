// Generated from the selected vehicle and battery profiles.
#pragma once
#include <array>
// X500 measured profile
namespace x500::profile {
inline constexpr const char *sha256 = "788b7bf0cc4194e8e4a61f39abbb7c28b1944893a706dba08ca7706be60b6219";
inline constexpr std::array<std::array<double,3>,4> positions{{{{0.174,-0.174,0.06}},{{-0.174,0.174,0.06}},{{0.174,0.174,0.06}},{{-0.174,-0.174,0.06}}}};
inline constexpr std::array<double,4> spins{1,1,-1,-1};
inline constexpr const char *esc_schema = "drone.esc.request/1";
inline constexpr const char *esc_order = "front_right,rear_left,front_left,rear_right";
inline constexpr unsigned series_cells = 4;
inline constexpr std::array<double,31> ocv_cell{2.5206077146360917,2.895944172810704,3.0498426795720754,3.1477551680774214,3.2267443693440594,3.294156556030869,3.3530786254333025,3.4588070773430677,3.5063613138428944,3.5346517338060917,3.5796791648975947,3.621543723481188,3.653933672193503,3.6835569420491794,3.715661977262932,3.7484474596926756,3.779617477262932,3.8085758568522743,3.834598684350936,3.8575728597720325,3.880111681395814,3.90699679380916,3.9429042138428945,3.9862801996394004,4.029531297052469,4.055929669051986,4.070186923076464,4.08019437922608,4.092534780817068,4.115956930280824,4.196523114636092};
inline constexpr double rotor_inertia = 2e-05;
inline constexpr double motor_ke = 0.010379670201645347;
inline constexpr double motor_resistance = 0.115;
inline constexpr double kt_prop = 1.2296996501129959e-05;
inline constexpr double kq_prop = 1.718657729631943e-07;
inline constexpr double friction = 4.1012149502659346e-17;
inline constexpr double coulomb = 0.008303736161277128;
inline constexpr double motor_limit = 40.0;
inline constexpr double dc_limit = 20.0;
inline constexpr double esc_efficiency = 0.858681252985813;
inline constexpr double motor_heat_capacity = 35.0;
inline constexpr double motor_cooling = 0.8;
inline constexpr double battery_heat_capacity = 805.6000000000001;
inline constexpr double battery_cooling = 0.35200000000000004;
inline constexpr double interconnect_resistance = 0.005;
inline constexpr double capacity_ah = 12.357353643836081;
inline constexpr double r0 = 0.0205139682902854;
inline constexpr double r1 = 0.018712556103200693;
inline constexpr double c1 = 295.1670831832436;
inline constexpr double auxiliary_a = 0.3;
inline constexpr double propeller_radius_m = 0.127;
inline constexpr double max_inplane_advance_ratio = 0.1;
inline constexpr double rotor_drag_coefficient = 8.06428e-05;
inline constexpr double body_drag_area = 0.08;
inline constexpr double max_air_speed_mps = 8.0;
inline constexpr double max_body_rate_radps = 2.0;
inline constexpr double max_rotor_speed_radps = 1000.0;
inline constexpr double min_soc = 0.2;
inline constexpr double min_pack_voltage_v = 12.4;
}
