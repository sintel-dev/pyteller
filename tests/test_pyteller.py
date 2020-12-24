# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# """Tests for `pyteller` package."""
#
# import unittest
#
# # from pyteller import pyteller
#
#
# class TestPyteller(unittest.TestCase):
#     """Tests for `pyteller` package."""
#
#     def setUp(self):
#         """Set up test fixtures, if any."""
#
#     def tearDown(self):
#         """Tear down test fixtures, if any."""
#
#     def test_000_something(self):
#         """Test something."""
#         self.assertTrue(True)
import pandas as pd
from pyteller.core import Pyteller
class TestPyteller:
    @classmethod
    def setup_class(cls):
        cls.clean = pd.DataFrame({
            'timestamp': list(range(100)),
            'value': [1] * 100,
        })

    def setup(self):
        self.pyteller = Pyteller(
                    pipeline='pyteller/pipelines/sandbox/persistence/persistence.json',
                    pred_length=6,
                    offset=1
                )

    def test_fit(self):
        self.pyteller.fit(self.clean)
