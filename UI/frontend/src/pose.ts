import { Euler, Matrix4, Quaternion, Vector3 } from 'three';
export function nedToScene(north: number, east: number, down: number) {
  return new Vector3(east, -down, -north);
}
export function attitudeQuaternion(roll: number, pitch: number, yaw: number) {
  const basis = new Matrix4().set(0,1,0,0, 0,0,-1,0, -1,0,0,0, 0,0,0,1);
  const body = new Matrix4().makeRotationFromEuler(new Euler(roll, pitch, yaw, 'ZYX'));
  const local = basis.clone().transpose();
  return new Quaternion().setFromRotationMatrix(basis.multiply(body).multiply(local));
}
