import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Compass, Focus } from 'lucide-react';
import type { Frame, Point } from './types';
import { attitudeQuaternion, nedToScene } from './pose';

export function Flight3D({ frame, points, connected }: {frame: Frame|null; points:Point[]; connected:boolean}) {
  const host=useRef<HTMLDivElement>(null);
  const live=useRef({frame,points,connected,follow:true});
  const [follow,setFollow]=useState(true), [failure,setFailure]=useState('');
  live.current={frame,points,connected,follow};
  useEffect(()=>{
    const el=host.current!;
    let renderer: THREE.WebGLRenderer;
    try { renderer=new THREE.WebGLRenderer({antialias:true,alpha:true}); }
    catch { setFailure('이 브라우저에서 3D 표시를 사용할 수 없습니다. 수치와 2D 궤적을 확인하세요.');return; }
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));el.appendChild(renderer.domElement);
    const scene=new THREE.Scene();scene.background=new THREE.Color('#f5f8fc');
    const camera=new THREE.PerspectiveCamera(48,1,.05,10000);camera.position.set(7,6,8);
    const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=1;controls.maxDistance=500;
    scene.add(new THREE.HemisphereLight('#ffffff','#b0c2d9',2.8));
    const light=new THREE.DirectionalLight('#ffffff',2);light.position.set(3,8,4);scene.add(light);
    const grid=new THREE.GridHelper(40,40,'#9cb7c8','#dce5ed');scene.add(grid);
    const model=new THREE.Group();scene.add(model);
    const material=new THREE.MeshStandardMaterial({color:'#176779',roughness:.6,transparent:true});
    const body=new THREE.Mesh(new THREE.BoxGeometry(.5,.15,.7),material);model.add(body);
    for (const angle of [-Math.PI/4,Math.PI/4]) {
      const arm=new THREE.Mesh(new THREE.BoxGeometry(1.6,.06,.07),material);arm.rotation.y=angle;model.add(arm);
    }
    for (const x of [-.55,.55]) for (const z of [-.55,.55]) {
      const rotor=new THREE.Mesh(new THREE.CylinderGeometry(.23,.23,.025,24),material);rotor.position.set(x,.10,z);model.add(rotor);
    }
    const noseMaterial=new THREE.MeshStandardMaterial({color:'#efac42',transparent:true});
    const nose=new THREE.Mesh(new THREE.ConeGeometry(.1,.25,3),noseMaterial);nose.rotation.x=-Math.PI/2;nose.position.set(0,.1,-.52);model.add(nose);
    const trail=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#0aa895',transparent:true,opacity:.75}));scene.add(trail);
    const resize=()=>{const {width,height}=el.getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/Math.max(1,height);camera.updateProjectionMatrix();};
    const observer=new ResizeObserver(resize);observer.observe(el);resize();
    let timer=0,lastRender=0,lastStamp='',lastPath='',lastRun='',lastTime=-1;
    const target=new THREE.Vector3();
    const draw=(time:number)=>{
      timer=requestAnimationFrame(draw);if(time-lastRender<1000/30)return;lastRender=time;
      const {frame:f,points:ps,connected:ok,follow:tracking}=live.current;
      const identity=f ? `${f.run_id}:${f.epoch}`:'';
      if(identity!==lastRun||(f&&f.t_ms<lastTime)){model.visible=false;model.position.set(0,0,0);model.quaternion.identity();target.set(0,0,0);lastRun=identity;lastPath='';lastStamp='';}
      if(f){
        const v=f.values;const keys=['north_m','east_m','height_m','roll_deg','pitch_deg','yaw_deg'];
        const valid=ok&&keys.every(k=>v[k]?.status==='valid'&&v[k]?.value!=null);
        material.opacity=valid?1:.24;noseMaterial.opacity=valid?1:.24;
        const stamp=`${identity}:${f.t_ms}:${ok}`;
        if(stamp!==lastStamp){
          if(valid){model.visible=true;
            model.position.copy(nedToScene(v.north_m.value!,v.east_m.value!,-v.height_m.value!));
            model.quaternion.copy(attitudeQuaternion(v.roll_deg.value!*Math.PI/180,v.pitch_deg.value!*Math.PI/180,v.yaw_deg.value!*Math.PI/180));
          }
          lastStamp=stamp;
        }
        const pathStamp=`${identity}:${f.t_ms}:${ps.length}`;
        if(pathStamp!==lastPath){
          const lines:number[]=[];let prev:Point|undefined;
          for(const p of ps){
            const vs=p.values;
            if(p.t_ms>f.t_ms||p.epoch!==f.epoch)continue;
            if(['north_m','east_m','height_m'].some(k=>vs[k]==null)){prev=undefined;continue;}
            if(prev&&p.t_ms-prev.t_ms<=1000){
              const a=nedToScene(prev.values.north_m!,prev.values.east_m!,-prev.values.height_m!);
              const b=nedToScene(vs.north_m!,vs.east_m!,-vs.height_m!);lines.push(...a.toArray(),...b.toArray());
            }
            prev=p;
          }
          trail.geometry.dispose();trail.geometry=new THREE.BufferGeometry();trail.geometry.setAttribute('position',new THREE.Float32BufferAttribute(lines,3));lastPath=pathStamp;
        }
        if(tracking){const delta=model.position.clone().sub(target);camera.position.add(delta);controls.target.copy(model.position);target.copy(model.position);}
        else target.copy(controls.target);
      }
      lastTime=f?.t_ms??-1;controls.update();renderer.render(scene,camera);
    };
    timer=requestAnimationFrame(draw);
    return()=>{cancelAnimationFrame(timer);observer.disconnect();controls.dispose();scene.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();}});material.dispose();noseMaterial.dispose();(trail.material as THREE.Material).dispose();renderer.dispose();renderer.domElement.remove();};
  },[]);
  const valid=connected&&frame&&['north_m','east_m','height_m','roll_deg','pitch_deg','yaw_deg'].every(k=>frame.values[k]?.status==='valid');
  return <section className="panel flight-3d"><div className="panel-heading"><div><span className="eyebrow">SPATIAL MONITOR</span><h2><Compass size={17}/>3D 비행 모니터</h2></div><button className={`button small ${follow?'active':''}`} onClick={()=>setFollow(!follow)} aria-pressed={follow}><Focus size={14}/>{follow?'기체 따라가기':'자유 시점'}</button></div>
    <div className="three-stage" ref={host} role="img" aria-label="수신한 자세와 로컬 위치를 표시하는 3D 드론">{failure&&<p className="three-fallback">{failure}</p>}</div>
    <div className="three-overlay"><span className={`pill ${valid?'teal':'neutral'}`}>{valid?'관측값 표시':'자세·위치 미수신 또는 만료'}</span><span>로컬 동·북 좌표 · 주황색은 기체 앞쪽</span></div>
    <div className="panel-footnote">드래그: 회전 · 스크롤: 확대 · 관측 표시 전용 · 물리 시뮬레이션 아님</div></section>;
}
