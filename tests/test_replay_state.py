from agent_rl.replay_state import build_replay_model, normalize_events


def test_normalize_preserves_order_and_description():
    raw = [
        {"turn": 2, "index": 1, "player_id": "RL", "buys": ["Silver"]},
        {"turn": 1, "index": 0, "player_id": "OPP", "buys": ["PASS"]},
    ]
    events = normalize_events(raw)
    assert [e.turn for e in events] == [1, 2]
    assert "passed" in events[0].readable_description


def test_frame_navigation_and_jump_to_turn():
    raw = [
        {"turn": 1, "index": 0, "player_id": "RL", "buys": ["Silver"]},
        {"turn": 1, "index": 1, "player_id": "OPP", "buys": ["PASS"]},
        {"turn": 2, "index": 2, "player_id": "RL", "buys": ["Gold"]},
    ]
    model = build_replay_model(raw)
    assert model.frame_count == 4
    assert model.next_index(0) == 1
    assert model.prev_index(0) == 0
    assert model.first_index_for_turn(2) == 3


def test_missing_optional_fields_do_not_crash():
    raw = [
        {"turn": 1, "player_id": "RL", "buys": ["ILLEGAL"]},
        {"turn": 2, "player_id": "RL"},
    ]
    model = build_replay_model(raw)
    assert model.frame_count == 3
    assert model.frame_at(999).frame_index == 2
    assert model.events[0].legal_actions is None
    assert model.events[0].state_diff is None


def test_invalid_turn_records_ignored():
    raw = [
        {"turn": "x", "player_id": "RL", "buys": ["Silver"]},
        {"turn": 3, "player_id": "RL", "buys": ["Gold"]},
    ]
    model = build_replay_model(raw)
    assert model.frame_count == 2
    assert model.frames[-1].turn == 3
