"""Official v2.1: typed fail-closed candidate gates. Invalid values are rejected."""
RISK_INPUTS=frozenset(['positive','negative','unknown','conflict'])
GATE_STATES=frozenset(['risk_hold','hold_conflicting_evidence','hold_unknown','pass_risk_gate_only'])
def gate(risk_evidence,quality_verified):
    if type(risk_evidence) is not str or risk_evidence not in RISK_INPUTS:raise ValueError('invalid risk evidence')
    if type(quality_verified) is not bool:raise TypeError('quality_verified must be a real bool')
    if risk_evidence=='conflict':return 'hold_conflicting_evidence'
    if risk_evidence=='positive':return 'risk_hold'
    if risk_evidence=='negative' and quality_verified:return 'pass_risk_gate_only'
    return 'hold_unknown'
def joint(skin_risk,eye_risk,skin_grade_ok,eye_grade_ok,skin_model_ok,eye_model_ok):
    if type(skin_risk) is not str or skin_risk not in GATE_STATES or type(eye_risk) is not str or eye_risk not in GATE_STATES:raise ValueError('invalid gate state')
    if any(type(x) is not bool for x in [skin_grade_ok,eye_grade_ok,skin_model_ok,eye_model_ok]):raise TypeError('all grade and model flags must be real booleans')
    if 'risk_hold' in (skin_risk,eye_risk):return 'not_joint_risk_hold'
    if skin_risk!='pass_risk_gate_only' or eye_risk!='pass_risk_gate_only':return 'not_joint_unknown'
    if not all([skin_grade_ok,eye_grade_ok,skin_model_ok,eye_model_ok]):return 'not_joint_incomplete_evidence'
    return 'joint_review_candidate_not_safety_approval'
