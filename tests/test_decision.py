import pytest

ALERT_CASES = [
    ("security", "A person is attempting to climb over the outer security barrier fence."),
    ("fire_disaster", "An active open flame is spreading across the trash bin with visible flickering fire."),
    ("nursing_care", "An elderly patient has tripped and fallen onto the floor near the bedside, lying motionless."),
    ("river_flood", "River water has breached the embankment dyke causing rapid inundation."),
    ("factory_safety", "A worker collapsed and is down unconscious inside the heavy machinery danger zone."),
    ("railway_platform", "A person has fallen onto active railway tracks beyond the yellow braille line."),
]

CALM_CASES = [
    ("security", "A person in standard work attire is walking through the entrance corridor normally."),
    ("fire_disaster", "The area is completely clear with normal workplace lighting and no visible smoke."),
    ("security", "Surveillance area appears calm and clear of any unusual activity."),
]


def test_all_six_presets_loaded(decision_engine):
    assert set(decision_engine.presets) == {
        "security", "fire_disaster", "nursing_care", "river_flood", "factory_safety", "railway_platform"
    }


@pytest.mark.parametrize("preset_id,state", ALERT_CASES)
def test_incident_text_raises_alert(decision_engine, preset_id, state):
    res = decision_engine.evaluate(state, preset_id)
    assert res["is_alert"], res
    assert res["score"] >= decision_engine.alert_threshold


@pytest.mark.parametrize("preset_id,state", CALM_CASES)
def test_calm_or_negated_text_does_not_alert(decision_engine, preset_id, state):
    res = decision_engine.evaluate(state, preset_id)
    assert not res["is_alert"], res


def test_unknown_preset_returns_error(decision_engine):
    assert "error" in decision_engine.evaluate("anything", "no_such_preset")


def test_choice_probabilities_are_normalized(decision_engine):
    res = decision_engine.evaluate("A suspicious person in a hoodie is loitering.", "security")
    answer = res["answers"]["action_type"]
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert answer["choice"] == "loitering"
