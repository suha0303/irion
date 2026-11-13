"""Pinned ESC_STATUS codec omitted by the installed pymavlink common dialect.

Class generated with pymavlink mavgen from PX4's common.xml, message 291.
PX4 mavlink submodule: 5bfd76d80281f6027134e854aafe6cb3dbfbe9e1.
Source: https://github.com/mavlink/mavlink/blob/5bfd76d80281f6027134e854aafe6cb3dbfbe9e1/message_definitions/v1.0/common.xml
Units are RPM, V, A; preserve signed RPM and standard checksum validation.
"""

import struct
from typing import Dict, Sequence

from pymavlink.dialects.v20.common import MAVLink, MAVLink_message

MAVLINK_MSG_ID_ESC_STATUS = 291


class MAVLink_esc_status_message(MAVLink_message):
    """
    ESC information for higher rate streaming. Recommended streaming
    rate is ~10 Hz. Information that changes more slowly is sent in
    ESC_INFO. It should typically only be streamed on high-bandwidth
    links (i.e. to a companion computer).
    """

    id = MAVLINK_MSG_ID_ESC_STATUS
    msgname = "ESC_STATUS"
    fieldnames = ["index", "time_usec", "rpm", "voltage", "current"]
    ordered_fieldnames = ["time_usec", "rpm", "voltage", "current", "index"]
    fieldtypes = ["uint8_t", "uint64_t", "int32_t", "float", "float"]
    fielddisplays_by_name: Dict[str, str] = {}
    fieldenums_by_name: Dict[str, str] = {}
    fieldunits_by_name: Dict[str, str] = {
        "time_usec": "us",
        "rpm": "rpm",
        "voltage": "V",
        "current": "A",
    }
    native_format = bytearray(b"<QiffB")
    orders = [4, 0, 1, 2, 3]
    lengths = [1, 4, 4, 4, 1]
    array_lengths = [0, 4, 4, 4, 0]
    crc_extra = 10
    unpacker = struct.Struct("<Q4i4f4fB")
    instance_field = "index"
    instance_offset = 56

    def __init__(
        self,
        index: int,
        time_usec: int,
        rpm: Sequence[int],
        voltage: Sequence[float],
        current: Sequence[float],
    ):
        MAVLink_message.__init__(
            self,
            MAVLink_esc_status_message.id,
            MAVLink_esc_status_message.msgname,
        )
        self._fieldnames = MAVLink_esc_status_message.fieldnames
        self._instance_field = MAVLink_esc_status_message.instance_field
        self._instance_offset = MAVLink_esc_status_message.instance_offset
        self.index = index
        self.time_usec = time_usec
        self.rpm = rpm
        self.voltage = voltage
        self.current = current

    def pack(self, mav: "MAVLink", force_mavlink1: bool = False) -> bytes:
        return self._pack(
            mav,
            self.crc_extra,
            self.unpacker.pack(
                self.time_usec,
                self.rpm[0],
                self.rpm[1],
                self.rpm[2],
                self.rpm[3],
                self.voltage[0],
                self.voltage[1],
                self.voltage[2],
                self.voltage[3],
                self.current[0],
                self.current[1],
                self.current[2],
                self.current[3],
                self.index,
            ),
            force_mavlink1=force_mavlink1,
        )
