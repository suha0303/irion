import { describe,expect,it } from 'vitest';
import { emptyProfile,fromLog,missingProfileFields,editProfile,editableProfileFields } from './LocalProfile';
import type { ProfileSuggestion } from './local-flight-types';

describe('log-derived profile form',()=>{
  it('asks only missing analysis values, not descriptive records',()=>{
    const p={...emptyProfile(),autopilot:'ardupilot',rotor_count:4,capacity_ah:44,output_min:1000,output_max:2000,imu_instance:1,battery_instance:0,actuator_instance:0,motor_channels:{front_right:1,front_left:3,rear_right:4,rear_left:2}};
    expect(missingProfileFields(p)).toEqual(['series_cells']);
    expect(missingProfileFields({...p,series_cells:12})).toEqual([]);
    expect(p.firmware).toBe('');expect(p.name).toBe('');
  });
  it('preserves unknowns and removes automatic provenance when manually changed',()=>{
    const evidence={value:44,source:'BATT_CAPACITY',source_sha256:'fixture',status:'logged_configuration'};
    const s:ProfileSuggestion={profile:{capacity_ah:44,provenance:{fields:{capacity_ah:evidence}}},source_sha256:'fixture',evidence:{capacity_ah:evidence},questions:[],health:'unknown',note:'',input_fields_present:true};
    const p=fromLog(s);
    expect(p.series_cells).toBeNull();expect(p.imu_instance).toBeNull();
    const edited=editProfile(p,'capacity_ah',22);
    expect(edited.capacity_ah).toBe(22);expect(edited.provenance?.fields?.capacity_ah).toBeUndefined();
    expect(p.provenance?.fields?.capacity_ah).toEqual(evidence);
  });
});


describe('editable cell estimate',()=>{
  it('keeps the cell field visible after autofill and preserves manual changes',()=>{
    const evidence={value:12,source:'voltage consistency',source_sha256:'fixture',status:'estimated'};
    const p={...emptyProfile(),series_cells:12,provenance:{fields:{series_cells:evidence}}};
    expect(editableProfileFields(p)).toContain('series_cells');
    const edited=editProfile(p,'series_cells',10);
    expect(edited.series_cells).toBe(10);
    expect(edited.provenance?.fields?.series_cells).toBeUndefined();
    expect(p.provenance.fields.series_cells.status).toBe('estimated');
    expect(editableProfileFields(edited)).toContain('series_cells');
  });
});
