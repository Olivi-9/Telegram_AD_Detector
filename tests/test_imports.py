"""Lightweight smoke tests for the project layout."""

from __future__ import annotations

import unittest


class ImportSmokeTest(unittest.TestCase):
    def test_core_modules_import(self) -> None:
        import ad_bot
        import bot_config
        import detection_engine
        import predict
        import telegram_ad_detector.cli
        import telegram_ad_detector.core.bot as core_bot
        import telegram_ad_detector.core.detection as core_detection
        import telegram_ad_detector.core.ml as core_ml
        import user_state

        self.assertTrue(callable(ad_bot.main))
        self.assertTrue(hasattr(bot_config, "config"))
        self.assertTrue(hasattr(detection_engine, "get_detection_engine"))
        self.assertTrue(callable(predict.main))
        self.assertTrue(callable(telegram_ad_detector.cli.bot))
        self.assertTrue(callable(core_bot.main))
        self.assertTrue(hasattr(core_detection, "get_detection_engine"))
        self.assertTrue(callable(core_ml.run_interactive))
        self.assertTrue(hasattr(user_state, "UserStateManager"))


if __name__ == "__main__":
    unittest.main()
