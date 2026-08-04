import unittest

from pipeline.utils.head_ablation import (
    QWEN3_14B_60_HEAD_ABLATION_CONFIG,
    qwen3_head_ablation,
    validate_qwen3_14b_head_config,
)
from run_harmbench_head_ablation_compliance import parse_judge_result, summarize


class HarmBenchAblationExperimentTests(unittest.TestCase):
    def test_config_contains_exactly_sixty_valid_heads(self):
        validate_qwen3_14b_head_config(QWEN3_14B_60_HEAD_ABLATION_CONFIG)
        self.assertEqual(sum(map(len, QWEN3_14B_60_HEAD_ABLATION_CONFIG.values())), 60)

    def test_judge_json_parsing(self):
        self.assertEqual(parse_judge_result('{"compliant": true, "reason": "actionable"}'), (True, "actionable", None))
        self.assertIsNotNone(parse_judge_result("not-json")[2])
        self.assertIsNotNone(parse_judge_result('{"compliant": "yes"}')[2])

    def test_summary_counts_only_valid_compliance_labels(self):
        rows = [{"judge": {"compliant": True, "error": None}}, {"judge": {"compliant": False, "error": "bad JSON"}}]
        self.assertEqual(summarize(rows), {"total_samples": 2, "compliant_count": 1, "compliance_rate": 0.5, "judge_errors": 1})

    def test_ablation_context_restores_forwards(self):
        class Attention:
            def forward(self):
                return "original"

        class Block:
            self_attn = Attention()

        class ModelBase:
            model_block_modules = [Block()]

        model_base = ModelBase()
        with qwen3_head_ablation(model_base, {0: [0]}):
            self.assertTrue(hasattr(model_base.model_block_modules[0].self_attn, "_head_indices_to_ablate"))
        self.assertEqual(model_base.model_block_modules[0].self_attn.forward(), "original")
        self.assertFalse(hasattr(model_base.model_block_modules[0].self_attn, "_head_indices_to_ablate"))
        with self.assertRaises(ValueError):
            with qwen3_head_ablation(model_base, {1: [0]}):
                pass
        self.assertEqual(model_base.model_block_modules[0].self_attn.forward(), "original")


if __name__ == "__main__":
    unittest.main()
