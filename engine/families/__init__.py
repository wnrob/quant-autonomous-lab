"""New research families (qlib models, arbitragelab pairs) wired into the SAME validation pipeline as ml_xs/xs_factor:
frozen Criteria (digest stored), store experiment dedup, DSR with up-front trial counting, mandatory risk gate, 3x cost stress.
Neither family can reach the protected test window and neither ever calls Store.record_test_access."""
