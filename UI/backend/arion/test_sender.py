"""Local synthetic stream sender. Never connects to a vehicle or control link."""
import argparse
import math
import socket
import time
import httpx
from pymavlink.dialects.v20 import common

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=14560)
    parser.add_argument('--seconds',type=float,default=45)
    parser.add_argument('--normalized-session')
    parser.add_argument('--dashboard-port',type=int,default=8765)
    args=parser.parse_args()
    mav=common.MAVLink(None,srcSystem=42,srcComponent=1)
    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    def send(message):
        sock.sendto(message.pack(mav),('127.0.0.1',args.port));mav.seq=(mav.seq+1)%256
    start=time.monotonic();seq=0
    with httpx.Client(base_url=f'http://127.0.0.1:{args.dashboard_port}',trust_env=False) as client:
        while time.monotonic()-start<args.seconds:
            t=time.monotonic()-start;seq+=1
            if args.normalized_session:
                values={'north_m':(3*math.sin(t/5),'m'),'east_m':(3*math.cos(t/5),'m'),'height_m':(2+.5*math.sin(t/3),'m'),
                    'roll_deg':(10*math.sin(t),'°'),'pitch_deg':(5*math.cos(t),'°'),'yaw_deg':(t*10%360,'°'),
                    'battery_voltage_v':(15.4,'V'),'battery_current_a':(24+math.sin(t),'A'),'battery_soc_pct':(82,'%')}
                r=client.post(f'/api/sessions/{args.normalized_session}/frames',json={'schema_version':'arion-normalized-v2',
                    'session_id':args.normalized_session,'sequence':seq,'epoch':0,'source_time_ns':time.monotonic_ns(),
                    'sample_period_ms':100,'preprocess_version':'normalized-v1','values':{
                        k:{'value':v,'unit':unit,'valid':True,'age_ms':0} for k,(v,unit) in values.items()}})
                r.raise_for_status()
            else:
                ms=int(t*1000)
                send(mav.attitude_encode(ms,.15*math.sin(t),.08*math.cos(t),t/6,0,0,1/6))
                send(mav.local_position_ned_encode(ms,3*math.sin(t/5),3*math.cos(t/5),-2-.5*math.sin(t/3),.2,.2,0))
                send(mav.esc_status_encode(int(t*1e6),0,[7000+int(100*math.sin(t+i)) for i in range(4)],[15.4]*4,[6.0]*4))
                if seq%2==0:
                    send(mav.battery_status_encode(0,0,0,32767,[3850]*4+[65535]*6,2450,-1,-1,82))
                if seq%5==0:send(mav.heartbeat_encode(2,12,128,4<<16|3<<24,4))
            time.sleep(.1)
    sock.close();print('시험 송신 완료 — 합성 관측, 실제 비행 아님')

if __name__=='__main__':main()
