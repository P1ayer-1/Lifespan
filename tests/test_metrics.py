"""Matrix arithmetic, hand-worked.

HAND is a 7x7 matrix written out below by hand. Every expected number in this
file was computed from it on paper and written in as a literal; none of them
came from running `metrics.py`.
"""

from __future__ import annotations

import pytest

from training import metrics
from training.config import N_PHASES

# M[i][j]: phase j's accuracy after training phase i. Diagonal is the peak,
# each column decays after its phase, off-diagonal upper triangle sits near the
# 0.25 chance floor.
#
#            j=0   j=1   j=2   j=3   j=4   j=5   j=6
HAND = [
    [0.80, 0.30, 0.25, 0.20, 0.25, 0.30, 0.25],  # i=0
    [0.70, 0.75, 0.30, 0.25, 0.25, 0.25, 0.30],  # i=1
    [0.60, 0.65, 0.85, 0.30, 0.25, 0.25, 0.25],  # i=2
    [0.55, 0.60, 0.70, 0.80, 0.30, 0.25, 0.25],  # i=3
    [0.50, 0.55, 0.65, 0.70, 0.75, 0.30, 0.25],  # i=4
    [0.45, 0.50, 0.60, 0.65, 0.70, 0.90, 0.30],  # i=5
    [0.40, 0.45, 0.55, 0.60, 0.65, 0.75, 0.80],  # i=6
]

UNTRAINED = [0.24, 0.26, 0.22, 0.28, 0.20, 0.30, 0.25]


def test_average_accuracy_is_the_mean_of_the_last_row():
    # last row: 0.40 + 0.45 + 0.55 + 0.60 + 0.65 + 0.75 + 0.80 = 4.20
    # 4.20 / 7 = 0.60
    assert metrics.average_accuracy(HAND, "continuation") == pytest.approx(0.60)


def test_per_phase_forgetting_is_column_peak_minus_final():
    # column peaks (max over i) and the final row:
    #   j=0 peak 0.80 (i=0), final 0.40 -> 0.40
    #   j=1 peak 0.75 (i=1), final 0.45 -> 0.30
    #   j=2 peak 0.85 (i=2), final 0.55 -> 0.30
    #   j=3 peak 0.80 (i=3), final 0.60 -> 0.20
    #   j=4 peak 0.75 (i=4), final 0.65 -> 0.10
    #   j=5 peak 0.90 (i=5), final 0.75 -> 0.15
    #   j=6 peak 0.80 (i=6), final 0.80 -> 0.00
    expected = [0.40, 0.30, 0.30, 0.20, 0.10, 0.15, 0.00]
    got = metrics.per_phase_forgetting(HAND, "continuation")
    assert len(got) == N_PHASES
    for g, e in zip(got, expected):
        assert g == pytest.approx(e)


def test_average_forgetting_is_the_mean_over_j_below_six():
    # (0.40 + 0.30 + 0.30 + 0.20 + 0.10 + 0.15) / 6 = 1.45 / 6 = 0.2416666667
    assert metrics.average_forgetting(HAND, "continuation") == pytest.approx(0.24166666667)


def test_forward_transfer_is_the_row_above_the_diagonal_minus_untrained():
    # j=1: M[0][1] 0.30 - 0.26 = 0.04
    # j=2: M[1][2] 0.30 - 0.22 = 0.08
    # j=3: M[2][3] 0.30 - 0.28 = 0.02
    # j=4: M[3][4] 0.30 - 0.20 = 0.10
    # j=5: M[4][5] 0.30 - 0.30 = 0.00
    # j=6: M[5][6] 0.30 - 0.25 = 0.05
    got = metrics.forward_transfer(HAND, UNTRAINED, "continuation")
    assert got[0] is None  # phase 0 has no earlier phase
    for g, e in zip(got[1:], [0.04, 0.08, 0.02, 0.10, 0.00, 0.05]):
        assert g == pytest.approx(e)


def test_fraction_of_peak_lost_is_h1s_number():
    # phase-0 column: peak 0.80 at i=0, final 0.40 -> (0.80 - 0.40) / 0.80 = 0.50
    # i.e. Arm A would have lost 50% of its peak phase-0 score: well past H1's
    # 15% confirmation threshold and nowhere near its 5% kill threshold.
    assert metrics.fraction_of_peak_lost(HAND, "continuation", 0) == pytest.approx(0.50)


def test_phase_column_is_the_score_after_each_phase():
    assert metrics.phase_column(HAND, 0, "continuation") == [
        0.80, 0.70, 0.60, 0.55, 0.50, 0.45, 0.40
    ]


# --------------------------------------------------------------------------- #
# the sign of perplexity, defined once and tested
# --------------------------------------------------------------------------- #

# A loss matrix: everything is 2.0 except phase 0's column, where the loss
# climbs from 1.0 after phase 0 to 1.6 after phase 6 - the model forgets.
PPL = [[2.0] * N_PHASES for _ in range(N_PHASES)]
for _i, _v in enumerate([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]):
    PPL[_i][0] = _v


def test_perplexity_is_negated_so_higher_is_better():
    assert metrics.to_score(1.0, "perplexity") == -1.0
    assert metrics.to_score(1.0, "cloze") == 1.0
    assert metrics.to_score(1.0, "continuation") == 1.0
    assert metrics.phase_column(PPL, 0, "perplexity")[0] == -1.0


def test_forgetting_on_perplexity_is_positive_when_the_loss_rose():
    # as scores, phase-0's column is -1.0 (peak, i=0) down to -1.6 (i=6),
    # so forgetting = -1.0 - (-1.6) = 0.60 nats per token.
    per = metrics.per_phase_forgetting(PPL, "perplexity")
    assert per[0] == pytest.approx(0.60)
    assert per[3] == pytest.approx(0.0)  # a flat column forgets nothing
    # average over j < 6: (0.60 + 0 + 0 + 0 + 0 + 0) / 6 = 0.10
    assert metrics.average_forgetting(PPL, "perplexity") == pytest.approx(0.10)


def test_average_accuracy_on_perplexity_is_the_negated_mean_loss():
    # last row: 1.6 + 2.0*6 = 13.6; 13.6 / 7 = 1.9428571; negated.
    assert metrics.average_accuracy(PPL, "perplexity") == pytest.approx(-1.94285714286)


def test_fraction_of_peak_lost_refuses_every_stored_only_key():
    """The frozen contract: `perplexity` (a loss, so a percentage of it means
    nothing) and `continuation_summed` (stored so the normalisation choice
    stays checkable) are never verdict metrics."""
    assert metrics.NEVER_HEADLINE == ("perplexity", "continuation_summed")
    assert metrics.VERDICT_EXAM_TYPES == ("cloze", "continuation")
    with pytest.raises(ValueError):
        metrics.fraction_of_peak_lost(PPL, "perplexity", 0)
    with pytest.raises(ValueError):
        metrics.fraction_of_peak_lost(HAND, "continuation_summed", 0)


def test_continuation_summed_is_stored_arithmetic_but_not_a_verdict():
    """The arithmetic accepts matrix.json's fourth key - it is an accuracy and
    higher is better - so the length-biased score can be reported beside the
    headline without ever deciding anything."""
    assert "continuation_summed" in metrics.STORED_EXAM_KEYS
    assert "continuation_summed" not in metrics.VERDICT_EXAM_TYPES
    assert metrics.to_score(0.4, "continuation_summed") == 0.4
    assert metrics.average_accuracy(HAND, "continuation_summed") == pytest.approx(0.60)
    assert metrics.average_forgetting(HAND, "continuation_summed") == pytest.approx(0.24166666667)


# --------------------------------------------------------------------------- #
# guards and units
# --------------------------------------------------------------------------- #


def test_points_are_percentage_points():
    assert metrics.to_points(0.05) == pytest.approx(5.0)
    assert metrics.to_points(0.03) == pytest.approx(3.0)


def test_unknown_exam_type_is_refused():
    with pytest.raises(ValueError):
        metrics.average_accuracy(HAND, "judge_grade")


def test_a_matrix_of_the_wrong_shape_is_refused():
    with pytest.raises(ValueError):
        metrics.validate_matrix(HAND[:6])
    with pytest.raises(ValueError):
        metrics.validate_matrix([row[:6] for row in HAND])


def test_forward_transfer_needs_a_seven_long_untrained_vector():
    with pytest.raises(ValueError):
        metrics.forward_transfer(HAND, UNTRAINED[:6], "continuation")


def test_fraction_of_peak_lost_refuses_a_non_positive_peak():
    zeroed = [[0.0] * N_PHASES for _ in range(N_PHASES)]
    with pytest.raises(ValueError):
        metrics.fraction_of_peak_lost(zeroed, "continuation", 0)


# --------------------------------------------------------------------------- #
# subset runs (train.py --phases 0,3,6): real ids, declared phases only
# --------------------------------------------------------------------------- #

NAN = float("nan")
SUBSET = (0, 3, 6)


def _subset_of(full, untrained):
    """What a 0/3/6 run writes: rows 1, 2, 4, 5 null, undeclared columns NaN,
    the declared cells taken from HAND."""
    M = [
        [full[i][j] if j in SUBSET else NAN for j in range(N_PHASES)] if i in SUBSET else [None] * N_PHASES
        for i in range(N_PHASES)
    ]
    u = [untrained[j] if j in SUBSET else NAN for j in range(N_PHASES)]
    return M, u


SUB, SUB_U = _subset_of(HAND, UNTRAINED)
# declared cells, read off HAND:
#            j=0   j=3   j=6
#   i=0     0.80  0.20  0.25
#   i=3     0.55  0.80  0.25
#   i=6     0.40  0.60  0.80
# untrained j=3 0.28, j=6 0.25


def test_subset_average_accuracy_is_the_last_declared_row_over_declared_phases():
    # (0.40 + 0.60 + 0.80) / 3 = 0.60
    assert metrics.average_accuracy(SUB, "continuation", phases=SUBSET) == pytest.approx(0.60)


def test_subset_forgetting_uses_declared_rows_and_never_renumbers():
    # j=0: max(0.80, 0.55, 0.40) - 0.40 = 0.40
    # j=3: max(0.20, 0.80, 0.60) - 0.60 = 0.20
    # j=6: max(0.25, 0.25, 0.80) - 0.80 = 0.00
    got = metrics.per_phase_forgetting(SUB, "continuation", phases=SUBSET)
    assert len(got) == N_PHASES
    assert [got[j] for j in (1, 2, 4, 5)] == [None] * 4
    assert got[0] == pytest.approx(0.40) and got[3] == pytest.approx(0.20) and got[6] == pytest.approx(0.0)
    # average over declared j except the last declared: (0.40 + 0.20) / 2
    assert metrics.average_forgetting(SUB, "continuation", phases=SUBSET) == pytest.approx(0.30)


def test_subset_forward_transfer_is_from_the_previous_declared_phase():
    # j=3: M[0][3] - untrained[3] = 0.20 - 0.28 = -0.08
    # j=6: M[3][6] - untrained[6] = 0.25 - 0.25 =  0.00
    got = metrics.forward_transfer(SUB, SUB_U, "continuation", phases=SUBSET)
    assert [got[j] for j in (0, 1, 2, 4, 5)] == [None] * 5
    assert got[3] == pytest.approx(-0.08) and got[6] == pytest.approx(0.0)


def test_subset_fraction_of_peak_lost_reads_phase_zero_over_declared_rows():
    # peak 0.80 (after 0), final 0.40 (after 6): 0.40 / 0.80 = 0.5
    assert metrics.fraction_of_peak_lost(SUB, "continuation", 0, phases=SUBSET) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="not one of the declared phases"):
        metrics.fraction_of_peak_lost(SUB, "continuation", 1, phases=SUBSET)


def test_subset_perplexity_is_negated_like_a_full_run():
    # Read SUB's cells as losses; score = -loss, so
    #   j=0: max(-0.80, -0.55, -0.40) - (-0.40) = 0.00
    #   j=3: max(-0.20, -0.80, -0.60) - (-0.60) = 0.40  (the loss rose 0.20 -> 0.60)
    got = metrics.per_phase_forgetting(SUB, "perplexity", phases=SUBSET)
    assert got[0] == pytest.approx(0.0) and got[3] == pytest.approx(0.40)


def test_subset_mode_refuses_a_missing_declared_cell_and_a_bad_phase_list():
    broken = [list(r) for r in SUB]
    broken[3][6] = None
    with pytest.raises(ValueError, match=r"M\[3\]\[6\]"):
        metrics.average_accuracy(broken, "continuation", phases=SUBSET)
    broken[3] = [None] * N_PHASES
    with pytest.raises(ValueError):
        metrics.average_forgetting(broken, "continuation", phases=SUBSET)
    for bad in ((3, 6), (0, 6, 3), (0, 7)):
        with pytest.raises(ValueError):
            metrics.average_accuracy(SUB, "continuation", phases=bad)
    u = list(SUB_U)
    u[3] = None
    with pytest.raises(ValueError, match=r"untrained\[3\]"):
        metrics.forward_transfer(SUB, u, "continuation", phases=SUBSET)


def test_a_full_run_gives_the_same_numbers_with_or_without_the_phase_list():
    """phases=None is the unchanged code path; declaring all seven must agree
    with it exactly, not merely approximately."""
    full = tuple(range(N_PHASES))
    for t in ("continuation", "perplexity"):
        assert metrics.average_accuracy(HAND, t, phases=full) == metrics.average_accuracy(HAND, t)
        assert metrics.average_forgetting(HAND, t, phases=full) == metrics.average_forgetting(HAND, t)
        assert metrics.per_phase_forgetting(HAND, t, phases=full) == metrics.per_phase_forgetting(HAND, t)
        assert metrics.forward_transfer(HAND, UNTRAINED, t, phases=full) == metrics.forward_transfer(
            HAND, UNTRAINED, t
        )
    assert metrics.fraction_of_peak_lost(HAND, "cloze", 0, phases=full) == metrics.fraction_of_peak_lost(
        HAND, "cloze", 0
    )
