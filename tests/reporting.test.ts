import {test} from 'node:test';import assert from 'node:assert/strict';import {groupOutcomeSummary} from '../dist/reporting.js';
const row=(sample_id:string,correct:boolean,status='judged')=>({qid:sample_id,sample_id,benchmark:'memops',status,correct});
const samples={A_remember:{benchmark:'memops',group_id:'A'},A_forget:{benchmark:'memops',group_id:'A'},B_remember:{benchmark:'memops',group_id:'B'}};
test('variants stay in their declared background and infrastructure failures stay in its denominator',()=>{
 const r=groupOutcomeSummary({planned_questions:3,sample_groups:samples},[row('A_remember',true),row('A_forget',false,'service_error'),row('B_remember',false)]);
 assert.equal(r.group_count,2);assert.deepEqual(r.groups,[{benchmark:'memops',group_id:'A',recorded:2,correct:1},{benchmark:'memops',group_id:'B',recorded:1,correct:0}]);assert.deepEqual(r.interval_95,[0,.5]);
});
test('one background cannot become several independent groups by adding operation variants',()=>{
 const r=groupOutcomeSummary({planned_questions:2,sample_groups:samples},[row('A_remember',true),row('A_forget',false)]);assert.equal(r.group_count,1);assert.equal(r.interval_95,null);assert.equal(r.unavailable_reason,'fewer_than_two_groups');
});
test('missing group metadata and unrecorded outcomes have no background confidence interval',()=>{
 const rows=[row('A_remember',true),row('B_remember',false)];
 assert.equal(groupOutcomeSummary({planned_questions:2},rows).unavailable_reason,'missing_group_metadata');
 const r=groupOutcomeSummary({planned_questions:3,sample_groups:samples},rows);assert.equal(r.interval_95,null);assert.equal(r.unavailable_reason,'unrecorded_questions');
});
