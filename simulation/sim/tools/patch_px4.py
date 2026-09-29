import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "sim/src"))
from capstone_sim.profile import load_profile  # noqa: E402

profile = load_profile()
p = root / "vendor/PX4-Autopilot"
g = p / "src/modules/simulation/gz_bridge"


def replace(path, old, new):
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"patch context not found: {path}")
    path.write_text(text.replace(old, new, 1))


shutil.copyfile(root / "sim/adapters/RawCapture.hpp", g / "RawCapture.hpp")
shutil.copyfile(root / "sim/adapters/CaptureTiming.hpp", g / "CaptureTiming.hpp")
replace(
    g / "GZBridge.hpp",
    '#include "GZGimbal.hpp"',
    '#include "GZGimbal.hpp"\n#include "RawCapture.hpp"\n#include <lib/battery/battery.h>',
)
replace(
    g / "GZBridge.hpp",
    "\tvoid Run() override;",
    "\tvoid Run() override;\n\tvoid batteryCallback(const gz::msgs::Double_V &msg);\n\tBattery _coupled_battery{1, this, 100000, battery_status_s::SOURCE_POWER_MODULE};\n\tRawCapture _raw_capture;",
)
replace(
    g / "GZBridge.cpp",
    "int GZBridge::init()\n{",
    'int GZBridge::init()\n{\n\tif (getenv("X500_CAPTURE_PATH")) {\n\t\t_node.Subscribe("/" + _model_name + "/battery_sensor", &GZBridge::batteryCallback, this);\n\t}',
)
callback = """void GZBridge::batteryCallback(const gz::msgs::Double_V &msg)
{
    if (msg.data_size() != 3 || !PX4_ISFINITE(msg.data(2)) || msg.data(2) < 0) { return; }
    _coupled_battery.setConnected(true);
    _coupled_battery.updateVoltage(msg.data(0));
    _coupled_battery.updateCurrent(msg.data(1));
    // Native battery timestamp is measurement time; exporter availability is copy time.
    const auto sample_time = static_cast<hrt_abstime>(llround(msg.data(2) * 1e6));
    _coupled_battery.updateBatteryStatus(sample_time);
    auto status = _coupled_battery.getBatteryStatus();
    status.timestamp = sample_time;
    _coupled_battery.publishBatteryStatus(status);
}

"""
bridge = (g / "GZBridge.cpp").read_text()
run_start = bridge.index("void GZBridge::Run()")
callback_start = bridge.find("void GZBridge::batteryCallback(")
if callback_start < 0:
    callback_start = run_start
(g / "GZBridge.cpp").write_text(bridge[:callback_start] + callback + bridge[run_start:])

replace(
    g / "GZBridge.cpp",
    "\tScheduleDelayed(10_ms);",
    "\t_raw_capture.update(hrt_absolute_time());\n\tScheduleDelayed(10_ms);",
)
replace(g / "CMakeLists.txt", "\t\tDEPENDS\n", "\t\tDEPENDS\n\t\t\tbattery\n")
replace(
    g / "GZMixingInterfaceESC.cpp",
    "if (!_node.Subscribe(motor_speed_topic,",
    'if (!getenv("X500_CAPTURE_PATH") && !_node.Subscribe(motor_speed_topic,',
)
capture_header = """        if (getenv("X500_CAPTURE_PATH")) {
            static uint64_t command_seq = 0;
            auto *header = rotor_velocity_message.mutable_header();
            const auto now = hrt_absolute_time();
            header->mutable_stamp()->set_sec(now / 1000000);
            header->mutable_stamp()->set_nsec((now % 1000000) * 1000);
            auto *schema = header->add_data();
            schema->set_key("schema"); schema->add_value(getenv("CAPSTONE_ESC_ORDER") ? "drone.esc.request/1" : "x500.esc.request/1");
            auto *order = header->add_data();
            order->set_key("order"); order->add_value(getenv("CAPSTONE_ESC_ORDER") ? getenv("CAPSTONE_ESC_ORDER") : "FR,RL,FL,RR");
            auto *semantic = header->add_data();
            semantic->set_key("semantic"); semantic->add_value("post_mapping_count_0_1000");
            auto *sequence = header->add_data();
            sequence->set_key("sequence"); sequence->add_value(std::to_string(command_seq++));
        }"""
if "static uint64_t command_seq = 0;" not in (g / "GZMixingInterfaceESC.cpp").read_text():
    declaration = "\t\tgz::msgs::Actuators rotor_velocity_message;"
    replace(g / "GZMixingInterfaceESC.cpp", declaration, declaration + "\n" + capture_header)

# The legacy X500 wire bytes remain the default. A selected generic profile supplies
# an explicit order; its differently sized arrays use a separate protocol identity.
replace(
    g / "GZMixingInterfaceESC.cpp",
    'schema->set_key("schema"); schema->add_value("x500.esc.request/1");',
    'schema->set_key("schema"); schema->add_value(getenv("CAPSTONE_ESC_ORDER") ? "drone.esc.request/1" : "x500.esc.request/1");',
)
replace(
    g / "GZMixingInterfaceESC.cpp",
    'order->set_key("order"); order->add_value("FR,RL,FL,RR");',
    'order->set_key("order"); order->add_value(getenv("CAPSTONE_ESC_ORDER") ? getenv("CAPSTONE_ESC_ORDER") : "FR,RL,FL,RR");',
)
# An earlier installer emitted this exact block with four extra leading spaces.
# Repair only that known variant; reviewed vendor hashes still guard all bytes.
replace(
    g / "GZMixingInterfaceESC.cpp",
    "\n".join("    " + line for line in capture_header.splitlines()),
    capture_header,
)

a = p / "ROMFS/px4fmu_common/init.d-posix/airframes/4001_gz_x500"
text = a.read_text()
text = text.split("# X500 coupled powertrain")[0].rstrip() + "\n"
if True:
    text += (
        '\n# X500 coupled powertrain: local simulation only\nif [ -n "$X500_CAPTURE_PATH" ]; then\n'
    )
    for key, val in {
        "SIM_BAT_ENABLE": 0,
        "BAT1_N_CELLS": 4,
        "BAT1_CAPACITY": 6000,
        "BAT1_R_INTERNAL": 0.01,
        "BAT1_V_EMPTY": 3.0,
        "BAT1_V_CHARGED": 4.15,
        "MPC_THR_HOVER": 0.5,
        "MPC_USE_HTE": 0,
    }.items():
        text += f"param set {key} {val}\n"
    for i, rotor in enumerate(profile["rotors"]):
        x, y, z = rotor["position_m"]
        text += f"param set CA_ROTOR{i}_PX {x}\nparam set CA_ROTOR{i}_PY {y}\nparam set CA_ROTOR{i}_PZ {z}\n"
        text += f"param set CA_ROTOR{i}_KM {rotor['spin'] * 0.05}\n"
    text += "fi\n"
    a.write_text(text)
