import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';
import { DecisionResult } from './LocalPreparation';

describe('aircraft residual decision display', () => {
  it('distinguishes missing calibration, normal range and suspected anomaly', () => {
    expect(renderToStaticMarkup(<DecisionResult/>)).toContain('판정 불가');
    expect(renderToStaticMarkup(<DecisionResult decision={{status:'unavailable',reason:'실측 대기'}}/>)).toContain('실측 대기');
    const normal = renderToStaticMarkup(<DecisionResult decision={{status:'normal',observations:4,exceedances:0,threshold:1,calibration_id:'1234567890'}}/>);
    expect(normal).toContain('정상 범위');
    expect(normal).toContain('실기체 성능 검증 대기');
    expect(normal).toContain('4개 중 0개');
    const suspect = renderToStaticMarkup(<DecisionResult decision={{status:'suspect',observations:4,exceedances:1,threshold:1}}/>);
    expect(suspect).toContain('이상 의심');
    expect(suspect).toContain('4개 중 1개');
  });
});

import { FlightEventResult } from './LocalPreparation';
it('keeps controller events distinct from aircraft health and primary core codes', () => {
  const markup=renderToStaticMarkup(<FlightEventResult events={{supported:true,health:'unknown',intervals:[],notes:[],error_counts:[
    {subsystem:5,code:1,count:9,label:'조종기 페일세이프'},
    {subsystem:24,code:0,count:1,label:'EKF 주 센서 변경'}]}}/>);
  expect(markup).toContain('건강 상태 미확정');
  expect(markup).toContain('발생');
  expect(markup).not.toContain('해제');
});
