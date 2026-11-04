import unittest
from risk_gates import gate,joint
class TestRisk(unittest.TestCase):
 def test_boolean_type_rejection(self):
  for bad in ['False','True','',0,1,None,[],{}]:
   with self.subTest(bad=bad):
    with self.assertRaises(TypeError):gate('negative',bad)
    with self.assertRaises(TypeError):joint('pass_risk_gate_only','pass_risk_gate_only',bad,True,True,True)
 def test_unknown_and_conflict_never_pass(self):
  self.assertEqual(gate('unknown',True),'hold_unknown');self.assertEqual(gate('conflict',True),'hold_conflicting_evidence')
  self.assertEqual(gate('negative',False),'hold_unknown')
 def test_positive_overrides_and_joint_separation(self):
  self.assertEqual(gate('positive',False),'risk_hold')
  self.assertEqual(joint('risk_hold','pass_risk_gate_only',True,True,True,True),'not_joint_risk_hold')
  self.assertEqual(joint('pass_risk_gate_only','pass_risk_gate_only',True,True,True,False),'not_joint_incomplete_evidence')
  self.assertEqual(joint('pass_risk_gate_only','pass_risk_gate_only',True,True,True,True),'joint_review_candidate_not_safety_approval')
 def test_invalid_evidence_rejected(self):
  for bad in [None,'',0,1,'False']:
   with self.assertRaises(ValueError):gate(bad,False)
  with self.assertRaises(ValueError):joint('pass','pass_risk_gate_only',True,True,True,True)
if __name__=='__main__':unittest.main(verbosity=2)
