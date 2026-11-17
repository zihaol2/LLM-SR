"""One final choice among frozen working-set candidates; no fits or references."""
import math
import json
from .memory import fit_version


def cards_for(archive):
    scale = archive.data.scale[archive.field]
    cards = []
    for state in archive.view():
        model = state['model']
        cards.append(dict(candidate=f"K{len(cards)+1:02d}", state=state['id'],
            model_version=fit_version(model), complexity=model['complexity'],
            mechanism_count=len(model['terms']), MSE=model['fit_mse'],
            normalized_MSE=model['fit_mse']/scale,
            normalized_RMSE=math.sqrt(model['fit_mse']/scale), fitted_terms=model['terms']))
    for card in cards:
        simpler = [other for other in cards if other['complexity'] < card['complexity']]
        other = max(simpler, key=lambda row: row['complexity']) if simpler else None
        card['versus_next_simpler'] = None if other is None else dict(
            candidate=other['candidate'], added_complexity=card['complexity']-other['complexity'],
            absolute_MSE_reduction=other['MSE']-card['MSE'],
            normalized_MSE_reduction=other['normalized_MSE']-card['normalized_MSE'],
            MSE_reduction_factor=other['MSE']/max(card['MSE'], 1e-300))
    return cards


RULES = '''Choose ONE existing frozen fitted equation per field to explain the observed time derivative.
The goal is a compact governing-equation hypothesis with adequate explanatory fit, not
merely a coarse low-complexity approximation to the observations.

Make two distinct comparisons:
1. Identify the compact explanatory core using the ENTIRE MSE/complexity frontier. A poor
   improvement at one intermediate candidate is not evidence that all later improvements
   are unimportant. Check every later compact candidate for renewed substantial gains.
2. Once a compact accurate core is identified, assess whether further fitted corrections
   justify their added complexity. Diminishing returns among the most accurate extensions
   support dropping those extensions; they do not justify discarding the compact accurate
   core itself and reverting to a much worse-fitting early candidate.

Use absolute MSE, normalized MSE, and especially normalized RMSE to assess fit adequacy.
Express normalized RMSE as a percentage of the target signal RMS. A numerically small raw
MSE can still be a large relative residual. Compare ALL supplied candidates rather than
stopping at the first apparent elbow. The nearest-neighbor comparison is only a convenience,
not a greedy stopping rule. Check your chosen candidate directly against the strongest
more-accurate compact alternative, not only the largest, lowest-MSE equation.

If rejecting a substantially more accurate compact candidate, provide concrete evidence
that its added mechanisms are unnecessary or numerically unreliable: inspect actual
expressions, fitted parameters, identifiable redundancy and compensating coefficients.
Calling an equation 'elaborate', unfamiliar or an 'ansatz' is not evidence of overfitting.
Spatial derivatives or coordinate dependence are not inherently implausible mechanisms.
Near-simple parameter values are suggestive, not proof of truth. A tiny coefficient alone
does not establish a negligible contribution; distinct derivative orders or powers do not
cancel merely because their coefficients have opposite signs.

No calibrated measurement-noise or derivative-error floor, uncertainty interval or
held-out prediction error is supplied. Missing calibration is uncertainty, not positive
evidence that a large fit improvement is noise. Do not assert any candidate lies below
a plausible noise floor or at numerical precision. Do not invent a universal error
threshold. A large MSE improvement alone also does not prove a mechanism is correct.

Simplicity matters after examining explanatory adequacy and the global tradeoff. Do not
automatically minimize MSE, complexity or any scalar Score. Prefer omitting unnecessary
correction terms when a simpler core already explains the observations comparably well.
If all choices are ambiguous, select the best supported candidate and describe uncertainty
without converting unsupported noise speculation into the reason for accepting poor fit.

MSE uses all observed times and the existing numerical spatial-window fitting objective;
it is not held-out prediction. No regional or time-bin residual tables are supplied.
No reference equation, benchmark name, accuracy label or previously selected answer is given.
Select an EXISTING candidate only; do not change coefficients, combine equations or request fits.
Return JSON only: {"selected":{"u":"Kxx"},"assessment":{"u":{
"complexity_MSE_tradeoff":"global comparison; identify compact explanatory core and later corrections",
"versus_lowest_MSE":"why extra correction terms are or are not worth their cost",
"versus_simpler":"fit adequacy in normalized RMSE percent and direct comparison with compact accurate alternatives",
"parameter_assessment":"concrete expression/parameter evidence; no unsupported overfitting labels",
"limitations":"uncertainty without asserting an unmeasured noise floor"}}}.
Use actual field names and candidate IDs.
'''


ASSESSMENT_KEYS=['complexity_MSE_tradeoff','versus_lowest_MSE','versus_simpler','parameter_assessment','limitations']


def selection_prompt(cards):
    # Generate a complete output schema from actual fields, rather than rely on
    # a scalar-u example for a coupled system. No answer labels are included.
    schema=dict(selected={field:'EXISTING_CANDIDATE_ID' for field in cards},
        assessment={field:{key:'required assessment' for key in ASSESSMENT_KEYS} for field in cards})
    return (RULES+'\nREQUIRED FIELD CONTRACT: Select exactly one candidate for EACH of '+
        json.dumps(list(cards))+'. Both selected and assessment must contain ALL these field keys. '+
        'Do not stop after the first field. Required output structure: '+json.dumps(schema,separators=(',',':'))+
        '\nCANDIDATES: '+json.dumps(cards,ensure_ascii=False,separators=(',',':')))


def validate(policy, cards):
    selected = policy.get('selected') if isinstance(policy, dict) else None
    assessment = policy.get('assessment') if isinstance(policy, dict) else None
    if not isinstance(selected, dict) or not isinstance(assessment, dict):
        raise ValueError('Final selection must provide selected and assessment field mappings')
    missing=set(cards)-(set(selected)&set(assessment))
    if missing:
        raise ValueError('Final selection is missing fields: '+', '.join(sorted(missing)))
    result = {}
    for field, rows in cards.items():
        row = next((r for r in rows if r['candidate'] == selected.get(field)), None)
        evidence = assessment.get(field)
        keys = ASSESSMENT_KEYS
        if row is None or not isinstance(evidence, dict) or not all(isinstance(evidence.get(k), str) and evidence[k].strip() for k in keys):
            raise ValueError('Unknown final candidate or missing quantitative assessment')
        result[field] = dict(state=row['state'], model_version=row['model_version'],
                             candidate=row['candidate'], assessment=evidence)
    return result
