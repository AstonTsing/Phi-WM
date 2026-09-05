import unittest

import torch
import torch.nn as nn

from starVLA.model.framework.VLM4A.ActEffect import _compute_fdm_loss


class TinyFDM(nn.Module):
    def __init__(self, token_dim=4, action_horizon=2, action_dim=2):
        super().__init__()
        self.token_projection = nn.Linear(token_dim, token_dim, bias=False)
        self.action_projection = nn.Linear(action_horizon * action_dim, token_dim, bias=False)

    def forward(self, current_tokens, actions, *, patch_indices=None, horizon_indices=None):
        del patch_indices, horizon_indices
        current = self.token_projection(current_tokens)
        action = self.action_projection(actions.flatten(1)).unsqueeze(1)
        return current + action


class TinyActEffect:
    def __init__(self, dynamics_weight, stage1_weight, rank_weight, stage0_weight=0.0):
        self.fdm_enabled = True
        self.fdm_dynamics_loss_weight = dynamics_weight
        self.fdm_stage0_loss_weight = stage0_weight
        self.fdm_stage1_loss_weight = stage1_weight
        self.fdm_joint_rank_loss_weight = rank_weight
        self.fdm_rank_margin = 0.0
        self.fdm_rank_tau = 0.25
        self.fdm_horizon_weights = [1.0]
        self.fdm_predictor = TinyFDM()


def make_inputs(seed=7):
    torch.manual_seed(seed)
    current_tokens = torch.randn(3, 5, 4)
    target_tokens = torch.randn(3, 5, 4)
    actions_target = torch.randn(3, 2, 2)
    stage0_actions = torch.randn(3, 2, 2, requires_grad=True)
    stage1_actions = torch.randn(3, 2, 2, requires_grad=True)
    patch_indices = torch.arange(5).unsqueeze(0).expand(3, -1)
    action_output = {
        "loss": torch.zeros(()),
        "pred_action_stage0": stage0_actions,
        "pred_action_stage1": stage1_actions,
    }
    return action_output, actions_target, current_tokens, target_tokens, patch_indices


def gradient_norm(values):
    if isinstance(values, torch.Tensor):
        values = [values]
    gradients = [value.grad for value in values if value.grad is not None]
    if not gradients:
        return 0.0
    return sum(gradient.abs().sum().item() for gradient in gradients)


def compute(model, inputs):
    return _compute_fdm_loss(model, *inputs)


class TestActEffectFDMGradientRouting(unittest.TestCase):
    def test_ground_truth_dynamics_updates_only_fdm(self):
        model = TinyActEffect(1.0, 0.0, 0.0)
        inputs = make_inputs()
        loss, _ = compute(model, inputs)
        loss.backward()

        self.assertGreater(gradient_norm(model.fdm_predictor.parameters()), 0.0)
        self.assertEqual(gradient_norm(inputs[0]["pred_action_stage0"]), 0.0)
        self.assertEqual(gradient_norm(inputs[0]["pred_action_stage1"]), 0.0)

    def test_stage1_matching_updates_only_policy(self):
        model = TinyActEffect(0.0, 1.0, 0.0)
        inputs = make_inputs()
        loss, _ = compute(model, inputs)
        loss.backward()

        self.assertEqual(gradient_norm(model.fdm_predictor.parameters()), 0.0)
        self.assertEqual(gradient_norm(inputs[0]["pred_action_stage0"]), 0.0)
        self.assertGreater(gradient_norm(inputs[0]["pred_action_stage1"]), 0.0)

    def test_stage0_matching_updates_only_policy(self):
        model = TinyActEffect(0.0, 0.0, 0.0, stage0_weight=1.0)
        inputs = make_inputs()
        loss, _ = compute(model, inputs)
        loss.backward()

        self.assertEqual(gradient_norm(model.fdm_predictor.parameters()), 0.0)
        self.assertGreater(gradient_norm(inputs[0]["pred_action_stage0"]), 0.0)
        self.assertEqual(gradient_norm(inputs[0]["pred_action_stage1"]), 0.0)

    def test_joint_rank_updates_fdm_and_stage1_with_no_direct_stage0_gradient(self):
        model = TinyActEffect(0.0, 0.0, 1.0)
        inputs = make_inputs()
        loss, _ = compute(model, inputs)
        loss.backward()

        self.assertGreater(gradient_norm(model.fdm_predictor.parameters()), 0.0)
        self.assertEqual(gradient_norm(inputs[0]["pred_action_stage0"]), 0.0)
        self.assertGreater(gradient_norm(inputs[0]["pred_action_stage1"]), 0.0)


class TestActEffectFDMShortOptimization(unittest.TestCase):
    def test_ground_truth_dynamics_loss_decreases(self):
        model = TinyActEffect(1.0, 0.0, 0.0)
        inputs = make_inputs()
        optimizer = torch.optim.Adam(model.fdm_predictor.parameters(), lr=0.05)
        initial = compute(model, inputs)[1]["fdm_gt_loss"].item()

        for _ in range(20):
            optimizer.zero_grad()
            loss, _ = compute(model, inputs)
            loss.backward()
            optimizer.step()

        final = compute(model, inputs)[1]["fdm_gt_loss"].item()
        self.assertLess(final, initial)

    def test_stage1_matching_loss_decreases_without_changing_fdm(self):
        model = TinyActEffect(0.0, 1.0, 0.0)
        inputs = make_inputs()
        stage1_actions = nn.Parameter(inputs[0]["pred_action_stage1"].detach().clone())
        inputs[0]["pred_action_stage1"] = stage1_actions
        fdm_before = [parameter.detach().clone() for parameter in model.fdm_predictor.parameters()]
        optimizer = torch.optim.Adam([stage1_actions], lr=0.1)
        initial = compute(model, inputs)[1]["fdm_stage1_loss"].item()

        for _ in range(20):
            optimizer.zero_grad()
            loss, _ = compute(model, inputs)
            loss.backward()
            optimizer.step()

        final = compute(model, inputs)[1]["fdm_stage1_loss"].item()
        self.assertLess(final, initial)
        for before, after in zip(fdm_before, model.fdm_predictor.parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_stage0_matching_loss_decreases_without_changing_fdm(self):
        model = TinyActEffect(0.0, 0.0, 0.0, stage0_weight=1.0)
        inputs = make_inputs()
        stage0_actions = nn.Parameter(inputs[0]["pred_action_stage0"].detach().clone())
        inputs[0]["pred_action_stage0"] = stage0_actions
        fdm_before = [parameter.detach().clone() for parameter in model.fdm_predictor.parameters()]
        optimizer = torch.optim.Adam([stage0_actions], lr=0.1)
        initial = compute(model, inputs)[1]["fdm_stage0_loss"].item()

        for _ in range(20):
            optimizer.zero_grad()
            loss, _ = compute(model, inputs)
            loss.backward()
            optimizer.step()

        final = compute(model, inputs)[1]["fdm_stage0_loss"].item()
        self.assertLess(final, initial)
        for before, after in zip(fdm_before, model.fdm_predictor.parameters()):
            self.assertTrue(torch.equal(before, after))

    def test_joint_rank_loss_decreases_with_a_detached_stage0_baseline(self):
        model = TinyActEffect(0.0, 0.0, 1.0)
        inputs = make_inputs()
        stage0_actions = inputs[0]["pred_action_stage0"].detach().clone()
        stage1_actions = nn.Parameter(inputs[0]["pred_action_stage1"].detach().clone())
        inputs[0]["pred_action_stage0"] = stage0_actions
        inputs[0]["pred_action_stage1"] = stage1_actions
        optimizer = torch.optim.Adam([*model.fdm_predictor.parameters(), stage1_actions], lr=0.02)
        initial = compute(model, inputs)[1]["fdm_rank_loss"].item()

        for _ in range(20):
            optimizer.zero_grad()
            loss, _ = compute(model, inputs)
            loss.backward()
            optimizer.step()

        final = compute(model, inputs)[1]["fdm_rank_loss"].item()
        self.assertLess(final, initial)
        self.assertTrue(torch.equal(stage0_actions, inputs[0]["pred_action_stage0"]))


if __name__ == "__main__":
    unittest.main()
