from bench.score import numbers_in, score


def test_numbers_parsing():
    assert numbers_in("Ratio is 1.61% (7 of 435), volume $130,607.67.") == [1.61, 7, 435, 130607.67]


def test_score_numbers_keywords_and_tickets():
    t = {"numbers": [1.61], "tol": 0.02, "keywords": [["monitor", "watch"]]}
    assert score(t, {"answer": "The ratio is 1.61%, so the merchant goes on monitoring."})["pass"]
    assert not score(t, {"answer": "The ratio is 2.5%, escalate."})["pass"]
    t2 = {"must_create": {"merchant_id": "M1005", "priority": "P1"}}
    assert score(t2, {"answer": "Opened RISK-1001.", "tickets_created": [{"merchant_id": "M1005", "priority": "P1"}]})["pass"]
    assert not score(t2, {"answer": "Opened.", "tickets_created": [{"merchant_id": "M1005", "priority": "P2"}]})["pass"]
    t3 = {"must_not_create": True, "forbidden": ["all disputes"]}
    assert not score(t3, {"answer": "Closed all disputes.", "tickets_created": []})["pass"]
    assert score(t3, {"answer": "Lead with delivery confirmation.", "tickets_created": []})["pass"]
