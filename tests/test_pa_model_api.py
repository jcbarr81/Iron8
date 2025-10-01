from fastapi.testclient import TestClient
from baseball_sim.serve.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_plate_appearance_minimal():
    req = {
      "game_id":"X",
      "state":{
        "inning":1,"half":"T","outs":0,
        "bases":{"1B":None,"2B":None,"3B":None},
        "score":{"away":0,"home":0},
        "count":{"balls":0,"strikes":0},
        "park_id":"GEN","weather":{"temp_f":70},"rules":{"dh":True}
      },
      "batter":{"player_id":"B1","bats":"R","ratings":{"pl":60,"gf":50,"fa":55}},
      "pitcher":{"player_id":"P1","throws":"L","ratings":{"arm":80,"fb":75,"sl":70,"fa":60}},
      "return_probabilities": True,
      "return_contact": False,
      "seed": 42
    }
    r = client.post("/v1/sim/plate-appearance", json=req)
    assert r.status_code == 200
    body = r.json()
    assert set(body["probs"].keys()) == {"BB","K","HBP","IP_OUT","1B","2B","3B","HR"}
    assert body["sampled_event"] in body["probs"]
