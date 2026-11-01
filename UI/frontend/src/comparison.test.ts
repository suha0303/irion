import { expect, it } from 'vitest';
import type { Detection } from './types';
import { compareModels } from './comparison';
const event:Detection={event_id:'e',provider_id:'lstm',run_id:'r',epoch:0,start_ms:26000,end_ms:28000,decided_at_ms:28200,valid_until_ms:32000,
  model_version:'fixture-v2',status:'suspect',target:'powertrain',score:.82,threshold:.65,title:'fixture',description:'',origin:'fixture',evidence:[],input_ref:{kind:'file',sha256:'abc'}};
it('identifies a disagreement for the same input window without comparing score magnitudes',()=>{
  expect(compareModels(event,{...event,provider_id:'transformer',status:'normal',score:500,threshold:1000}).state).toBe('different');
});
it('does not call two different targets a model disagreement',()=>{
  expect(compareModels(event,{...event,target:'battery',status:'normal'}).state).toBe('unmatched');
});
it('does not compare different windows, epochs or source hashes',()=>{
  for(const change of [{start_ms:25000},{epoch:1},{input_ref:{kind:'file',sha256:'other'}}])expect(compareModels(event,{...event,...change}).state).toBe('unmatched');
});
it('keeps missing and unavailable models out of normal/agreed states',()=>{
  expect(compareModels(null,event).state).toBe('waiting');
  expect(compareModels(event,{...event,status:'unavailable'}).state).toBe('waiting');
});
it('reports matching valid decisions as agreement',()=>{
  expect(compareModels(event,{...event,provider_id:'transformer'}).state).toBe('agree');
});
