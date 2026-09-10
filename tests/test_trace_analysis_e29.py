"""Small synthetic fail-closed boundaries; no model dependencies."""
import copy
import unittest
from scripts.trace_analysis_e29 import diagnose, features, index, metrics, overlap, risk_table, trace

class Boundaries(unittest.TestCase):
    def sample(self):
        p=('ADD','SWAP','XOR','SWAP','ADD'); state=(15,1)
        t=trace(p,state)
        item=dict(state=list(state),target_trace=t,predicted_trace=copy.deepcopy(t),prefix_joint_correct=[True]*5,joint_final_correct=True,stratum='test')
        return p,item,{state:'test'}

    def test_recovery_and_risk(self):
        p,item,m=self.sample()
        item['predicted_trace'][1]=[0,0]; item['prefix_joint_correct'][1]=False
        r=diagnose(p,item,m)
        self.assertEqual(r['first'],1); self.assertTrue(r['recovery'])
        self.assertEqual(metrics([r])['full_trace'],0)
        rows=risk_table([(p,r)])
        at=lambda i,op:next(x for x in rows if x['position']==i and x['opcode']==op)
        self.assertEqual(at(2,'SWAP')['first_errors'],1)
        self.assertEqual(at(3,'XOR')['exposed'],0)
        self.assertIsNone(at(3,'XOR')['rate'])

    def test_semantics(self):
        self.assertEqual(trace(('ADD','XOR','SWAP'),(15,1)),[[0,1],[1,1],[1,1]])
        self.assertEqual(features('ADD',(15,1)),dict(x_equal_y=False,either_zero=False,overflow=True,carry_present=True))
        self.assertEqual(features('XOR',(0,15))['xor_popcount'],4)

    def test_identity_alignment(self):
        rows=[dict(state=[2,3]),dict(state=[0,1])]
        self.assertEqual(index(rows,lambda r:tuple(r['state'])),index(list(reversed(rows)),lambda r:tuple(r['state'])))
        with self.assertRaises(ValueError): index(rows+rows[:1],lambda r:tuple(r['state']))

    def test_malformed_and_wrong_identity(self):
        for field,value in [('target_trace',[[0,0]]*5),('stratum','train'),('predicted_trace',[[True,1]]*5),('prefix_joint_correct',[1]*5)]:
            p,item,m=self.sample();item[field]=value
            with self.assertRaises(ValueError): diagnose(p,item,m)

    def test_empty_overlap(self):
        self.assertEqual(overlap([set(),set()]),dict(intersection=0,union=0,jaccard=None))
        self.assertEqual(overlap([{1,2},{2,3}])['intersection'],1)

    def test_equal_swap_not_wrong_identity(self):
        p=('SWAP',)*5
        item=dict(state=[3,3],target_trace=[[3,3]]*5,predicted_trace=[[3,3]]*5,prefix_joint_correct=[True]*5,joint_final_correct=True,stratum='test')
        rows=risk_table([(p,diagnose(p,item,{(3,3):'test'}))])
        s=next(r for r in rows if r['position']==1 and r['opcode']=='SWAP')['swap_patterns']['equal_inputs']
        self.assertEqual(s,dict(exposed=1,correct_swap=1,wrong_identity=0,other_wrong=0))

if __name__=='__main__': unittest.main()
