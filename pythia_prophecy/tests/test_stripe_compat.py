"""Regression test for the stripe v15 object-shape break (2026-09-19).

    cd pythia_prophecy && .venv/bin/python tests/test_stripe_compat.py

Runs against the real installed stripe SDK (pinned 15.x) plus synthetic pre-v15
and mapping shapes, so _stripe_to_dict is proven on every shape it may see.
"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault("JWT_SECRET_KEY", "test-only")
import stripe
from stripe import StripeObject
from api.service import _stripe_to_dict

_passed = []
def check(name, cond):
    if not cond: raise AssertionError("FAIL: " + name)
    _passed.append(name)

nested = {"id": "sub_1", "status": "canceled", "customer": "cus_1", "n": None,
          "items": {"data": [{"id": "si_1", "price": {"id": "price_1"}}]}}
obj = StripeObject.construct_from(nested, "sk_test_x")
check("v15 StripeObject has no .get (the trigger)", not hasattr(obj, "get"))
d = _stripe_to_dict(obj)
check("returns a plain dict", type(d) is dict)
check("nested list items are plain dicts", type(d["items"]["data"][0]) is dict)
check("deeply nested is plain", type(d["items"]["data"][0]["price"]) is dict)
check(".get works after conversion", d.get("status") == "canceled" and d.get("missing") is None)
check("None preserved", d["n"] is None)

ev = stripe.Event.construct_from({"id": "evt_1", "type": "customer.subscription.deleted",
                                  "data": {"object": nested}}, "sk_test_x")
e = _stripe_to_dict(ev)
check("Event converts (the webhook line that raised KeyError: 0)", e["type"] == "customer.subscription.deleted")
check("event data.object is plain", type(e["data"]["object"]) is dict and e["data"]["object"]["id"] == "sub_1")

lst = stripe.ListObject.construct_from({"object": "list", "data": [nested], "has_more": False}, "sk_test_x")
l = _stripe_to_dict(lst)
check("ListObject -> dict with data list", isinstance(l.get("data"), list) and l["data"][0]["id"] == "sub_1")

check("plain dict passthrough", _stripe_to_dict(nested) is nested)
check("None -> {}", _stripe_to_dict(None) == {})

class OldStyle:  # pre-v15 shape
    def to_dict_recursive(self): return {"id": "old", "k": {"v": 1}}
check("pre-v15 to_dict_recursive supported", _stripe_to_dict(OldStyle())["k"]["v"] == 1)

class Mapping:  # generic mapping without to_dict
    def keys(self): return ["a"]
    def __getitem__(self, k): return {"a": 1}[k]
check("generic mapping supported", _stripe_to_dict(Mapping()) == {"a": 1})

print(f"\nAll {len(_passed)} checks passed:"); [print("  PASS", n) for n in _passed]
