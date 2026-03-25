import numpy as np
import pandas as pd
import pytest
from playerank.models.Rater import Rater


def test_get_rating_no_goals():
    rater = Rater(alpha_goal=0.0)
    assert rater.get_rating(5.0, 2) == pytest.approx(5.0)


def test_get_rating_goals_only():
    rater = Rater(alpha_goal=1.0)
    assert rater.get_rating(5.0, 2) == pytest.approx(2.0)


def test_get_rating_mixed():
    rater = Rater(alpha_goal=0.5)
    assert rater.get_rating(4.0, 2.0) == pytest.approx(3.0)


def test_predict_normalized_range():
    rater = Rater(alpha_goal=0.0)
    df = pd.DataFrame({
        "playerankScore": [1.0, 2.0, 3.0, 4.0],
        "goalScored": [0, 0, 0, 0],
    })
    ratings = rater.predict(df, goal_feature="goalScored", score_feature="playerankScore")
    assert ratings.min() == pytest.approx(0.0)
    assert ratings.max() == pytest.approx(1.0)


def test_predict_output_length():
    rater = Rater(alpha_goal=0.0)
    df = pd.DataFrame({
        "playerankScore": [1.0, 2.0, 3.0],
        "goalScored": [1, 0, 2],
    })
    ratings = rater.predict(df, goal_feature="goalScored", score_feature="playerankScore")
    assert len(ratings) == 3


def test_predict_goals_affect_order():
    # With alpha=1.0 only goals matter; ranking should follow goal count
    rater = Rater(alpha_goal=1.0)
    df = pd.DataFrame({
        "playerankScore": [100.0, 100.0, 100.0],
        "goalScored": [0.0, 1.0, 2.0],
    })
    ratings = rater.predict(df, goal_feature="goalScored", score_feature="playerankScore")
    assert ratings[0] < ratings[1] < ratings[2]


def test_predict_equal_scores_equal_ratings():
    rater = Rater(alpha_goal=0.0)
    df = pd.DataFrame({
        "playerankScore": [3.0, 3.0, 3.0],
        "goalScored": [0, 0, 0],
    })
    ratings = rater.predict(df, goal_feature="goalScored", score_feature="playerankScore")
    assert ratings[0] == pytest.approx(ratings[1])
    assert ratings[1] == pytest.approx(ratings[2])
