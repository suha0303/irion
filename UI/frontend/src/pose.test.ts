import {expect,it} from 'vitest';
import {Vector3} from 'three';
import {attitudeQuaternion,nedToScene} from './pose';
const close=(a:Vector3,b:number[])=>a.toArray().forEach((v,i)=>expect(v).toBeCloseTo(b[i],8));
it('maps north east and ascent to scene axes',()=>{close(nedToScene(1,2,-3),[2,3,-1]);});
it('faces north at zero attitude and east at yaw 90 degrees',()=>{
 close(new Vector3(0,0,-1).applyQuaternion(attitudeQuaternion(0,0,0)),[0,0,-1]);
 close(new Vector3(0,0,-1).applyQuaternion(attitudeQuaternion(0,0,Math.PI/2)),[1,0,0]);
});
it('positive pitch raises the nose; positive roll lowers the right arm',()=>{
 close(new Vector3(0,0,-1).applyQuaternion(attitudeQuaternion(0,Math.PI/2,0)),[0,1,0]);
 close(new Vector3(1,0,0).applyQuaternion(attitudeQuaternion(Math.PI/2,0,0)),[0,-1,0]);
});
