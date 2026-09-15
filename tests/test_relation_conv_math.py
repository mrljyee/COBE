from __future__ import annotations

import io
import math

import torch

from stage1_architectures import (
    PaperDualUNet1D,
    PaperRelationDualUNet1D,
    PaperWidthDualUNet1D,
    count_trainable_parameters,
    initialize_relation_backbone_from_paper,
)
from stage1_relation_layers import (
    NonlocalPolynomialRelation1D,
    RelationConv1D,
    chebyshev_basis,
)


ATOL = 1e-6


def assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = ATOL) -> None:
    difference = float((actual - expected).abs().max().detach().cpu())
    if not difference <= atol:
        raise AssertionError(f"max_abs_difference={difference:.3e} exceeds atol={atol:.3e}")


def test_chebyshev_values() -> None:
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    basis = chebyshev_basis(x, 4)
    expected = (
        torch.ones_like(x),
        x,
        2 * x**2 - 1,
        4 * x**3 - 3 * x,
        8 * x**4 - 8 * x**2 + 1,
    )
    for actual, target in zip(basis, expected):
        assert_close(actual, target)


def test_attention_and_mask() -> None:
    torch.manual_seed(11)
    layer = RelationConv1D(
        3,
        8,
        kernel_size=5,
        padding=2,
        relation_dim=8,
        num_heads=2,
        degree=3,
        nonlocal_radius=2,
    ).eval()
    x = torch.randn(2, 3, 17)
    output, diagnostics = layer(x, return_diagnostics=True)
    if output.shape != (2, 8, 17):
        raise AssertionError(f"Unexpected output shape: {tuple(output.shape)}")
    row_sums = diagnostics.attention.sum(dim=-1)
    assert_close(row_sums, torch.ones_like(row_sums))
    for i in range(17):
        for j in range(17):
            if 0 < abs(i - j) <= 2:
                if not torch.equal(
                    diagnostics.attention[..., i, j],
                    torch.zeros_like(diagnostics.attention[..., i, j]),
                ):
                    raise AssertionError(f"Local neighbour ({i},{j}) was not excluded.")
    if not torch.isfinite(output).all():
        raise AssertionError("Non-finite output detected.")


def test_factorized_relation_matches_explicit_pairs() -> None:
    torch.manual_seed(12)
    module = NonlocalPolynomialRelation1D(
        channels=4,
        relation_dim=4,
        num_heads=2,
        degree=3,
        nonlocal_radius=1,
        max_relative_distance=16,
    ).double().eval()
    with torch.no_grad():
        module.coefficients[:, 1, 1] = torch.tensor([0.7, -0.2], dtype=torch.float64)
        module.coefficients[:, 1, 2] = torch.tensor([0.3, 0.4], dtype=torch.float64)
        module.coefficients[:, 2, 1] = torch.tensor([-0.1, 0.5], dtype=torch.float64)
    features = torch.randn(1, 4, 7, dtype=torch.float64)
    efficient, attention = module(features, return_attention=True)

    x = module.norm(features.transpose(1, 2))
    a = torch.tanh(module._split_heads(module.left(x)))
    b = torch.tanh(module._split_heads(module.right(x)))
    basis_a = chebyshev_basis(a, module.degree)
    basis_b = chebyshev_basis(b, module.degree)
    explicit_pairs = torch.zeros(
        1,
        module.num_heads,
        7,
        7,
        module.head_dim,
        dtype=torch.float64,
    )
    for p in range(1, module.degree + 1):
        for q_degree in range(1, module.degree + 1):
            if not bool(module.pair_mask[p, q_degree]):
                continue
            coefficient = module.coefficients[:, p, q_degree][None, :, None, None, None]
            explicit_pairs = explicit_pairs + coefficient * (
                basis_a[p].unsqueeze(3) * basis_b[q_degree].unsqueeze(2)
            )
    explicit = (attention.unsqueeze(-1) * explicit_pairs).sum(dim=3)
    explicit = explicit.transpose(1, 2).reshape(1, 7, module.relation_dim)
    explicit = module.output(explicit).transpose(1, 2)
    assert_close(efficient, explicit, atol=1e-10)


def test_zero_gate_is_exact_local_convolution() -> None:
    torch.manual_seed(13)
    layer = RelationConv1D(
        3,
        8,
        relation_dim=8,
        num_heads=2,
        degree=3,
        gate_init=0.0,
    ).eval()
    x = torch.randn(2, 3, 23)
    output, diagnostics = layer(x, return_diagnostics=True)
    assert_close(output, diagnostics.local, atol=0.0)
    if float(diagnostics.gate.detach()) != 0.0:
        raise AssertionError("Zero gate did not produce an exact zero multiplier.")


def test_zero_gate_can_wake_up() -> None:
    torch.manual_seed(14)
    layer = RelationConv1D(
        2,
        8,
        relation_dim=8,
        num_heads=2,
        degree=3,
        gate_init=0.0,
    )
    x = torch.randn(2, 2, 19)
    output, diagnostics = layer(x, return_diagnostics=True)
    loss = (output * diagnostics.relation.detach()).sum()
    loss.backward()
    gate_gradient = float(layer.gate.grad.detach())
    if not math.isfinite(gate_gradient) or abs(gate_gradient) <= 1e-10:
        raise AssertionError(f"Zero-initialized gate cannot wake up: grad={gate_gradient}")


def test_relation_parameters_receive_gradients_when_gate_opens() -> None:
    torch.manual_seed(15)
    layer = RelationConv1D(
        2,
        8,
        relation_dim=8,
        num_heads=2,
        degree=3,
        gate_init=0.2,
    )
    x = torch.randn(2, 2, 19)
    loss = layer(x).square().mean()
    loss.backward()
    required = {
        "query": layer.relation.query.weight.grad,
        "key": layer.relation.key.weight.grad,
        "left": layer.relation.left.weight.grad,
        "right": layer.relation.right.weight.grad,
        "coefficients": layer.relation.coefficients.grad,
        "output": layer.relation.output.weight.grad,
    }
    for name, gradient in required.items():
        if gradient is None or not torch.isfinite(gradient).all():
            raise AssertionError(f"Missing or non-finite gradient for {name}.")
        if float(gradient.abs().sum()) <= 0.0:
            raise AssertionError(f"Zero gradient for {name}.")
    active_gradient = layer.relation.coefficients.grad[layer.relation.pair_mask[None].expand_as(
        layer.relation.coefficients
    )]
    if float(active_gradient.abs().sum()) <= 0.0:
        raise AssertionError("Active polynomial coefficients received no gradient.")


def test_gate_gradient_matches_finite_difference() -> None:
    torch.manual_seed(16)
    layer = RelationConv1D(
        2,
        4,
        relation_dim=4,
        num_heads=2,
        degree=3,
        gate_init=0.17,
    ).double().eval()
    x = torch.randn(1, 2, 9, dtype=torch.float64)
    loss = layer(x).square().mean()
    loss.backward()
    analytical = float(layer.gate.grad)
    original = float(layer.gate.detach())
    epsilon = 1e-6
    with torch.no_grad():
        layer.gate.fill_(original + epsilon)
        plus = float(layer(x).square().mean())
        layer.gate.fill_(original - epsilon)
        minus = float(layer(x).square().mean())
        layer.gate.fill_(original)
    numerical = (plus - minus) / (2 * epsilon)
    if abs(analytical - numerical) > 1e-6:
        raise AssertionError(
            f"Gate gradient mismatch: analytical={analytical:.9g}, numerical={numerical:.9g}"
        )


def _copy_baseline_feature_weights(
    baseline: PaperDualUNet1D,
    relation_model: PaperRelationDualUNet1D,
) -> None:
    for branch_name in ("real_net", "imag_net"):
        baseline_branch = getattr(baseline, branch_name)
        relation_branch = getattr(relation_model, branch_name)
        for layer_name in ("enc1", "enc2", "bottleneck", "dec2", "dec1"):
            source = getattr(baseline_branch, layer_name)[0]
            target = getattr(relation_branch, layer_name).local_conv
            target.load_state_dict(source.state_dict())
        relation_branch.out.load_state_dict(baseline_branch.out.state_dict())


def test_full_dual_unet_nests_original_at_zero_gate() -> None:
    torch.manual_seed(17)
    baseline = PaperDualUNet1D(in_channels=1).eval()
    relation_model = PaperRelationDualUNet1D(
        in_channels=1,
        relation_dim=8,
        num_heads=2,
        degree=3,
        nonlocal_radius=2,
        gate_init=0.0,
    ).eval()
    _copy_baseline_feature_weights(baseline, relation_model)
    x = torch.randn(1, 1, 451)
    with torch.no_grad():
        baseline_output = baseline(x)
        relation_output = relation_model(x)
    if baseline_output.shape != (1, 2, 451):
        raise AssertionError(f"Unexpected baseline shape: {tuple(baseline_output.shape)}")
    assert_close(relation_output, baseline_output, atol=0.0)


def test_seed_matched_backbone_initialization() -> None:
    torch.manual_seed(171)
    baseline = PaperDualUNet1D(in_channels=1).eval()
    torch.manual_seed(171)
    matched_backbone = PaperDualUNet1D(in_channels=1)
    relation_model = PaperRelationDualUNet1D(
        in_channels=1,
        relation_dim=8,
        num_heads=2,
        degree=3,
        gate_init=0.0,
    ).eval()
    initialize_relation_backbone_from_paper(relation_model, matched_backbone)
    x = torch.randn(1, 1, 451)
    with torch.no_grad():
        assert_close(relation_model(x), baseline(x), atol=0.0)


def test_relation_layer_shapes_for_all_unet_sites() -> None:
    sites = [
        (1, 16, 451),
        (16, 32, 225),
        (32, 64, 112),
        (96, 32, 225),
        (48, 16, 451),
    ]
    with torch.no_grad():
        for input_channels, output_channels, length in sites:
            layer = RelationConv1D(
                input_channels,
                output_channels,
                relation_dim=min(8, output_channels),
                num_heads=2,
                degree=3,
            ).eval()
            output = layer(torch.randn(1, input_channels, length))
            expected_shape = (1, output_channels, length)
            if output.shape != expected_shape:
                raise AssertionError(
                    f"Site {(input_channels, output_channels, length)} produced {tuple(output.shape)}"
                )


def test_state_dict_round_trip() -> None:
    torch.manual_seed(18)
    source = RelationConv1D(2, 8, relation_dim=8, num_heads=2, degree=3).eval()
    x = torch.randn(1, 2, 13)
    expected = source(x)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    target = RelationConv1D(2, 8, relation_dim=8, num_heads=2, degree=3).eval()
    target.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    assert_close(target(x), expected, atol=0.0)


def test_relation_ablation_modes() -> None:
    torch.manual_seed(181)
    x = torch.randn(1, 3, 17)
    outputs: dict[str, torch.Tensor] = {}
    diagnostics = {}
    for mode in (
        "joint",
        "attention_only",
        "polynomial_only",
        "shuffled_joint",
        "uniform_global",
        "position_only",
    ):
        layer = RelationConv1D(
            3,
            8,
            relation_dim=8,
            num_heads=2,
            degree=3,
            relation_mode=mode,
            gate_init=0.2,
        ).eval()
        output, diagnostic = layer(x, return_diagnostics=True)
        if output.shape != (1, 8, 17) or not torch.isfinite(output).all():
            raise AssertionError(f"Invalid output for relation mode {mode}.")
        outputs[mode] = output
        diagnostics[mode] = diagnostic
    identity = torch.eye(17)[None, None].expand(1, 2, 17, 17)
    assert_close(diagnostics["polynomial_only"].attention, identity, atol=0.0)

    joint = RelationConv1D(
        3,
        8,
        relation_dim=8,
        num_heads=2,
        degree=3,
        relation_mode="joint",
        gate_init=0.2,
    ).eval()
    shuffled = RelationConv1D(
        3,
        8,
        relation_dim=8,
        num_heads=2,
        degree=3,
        relation_mode="shuffled_joint",
        gate_init=0.2,
    ).eval()
    shuffled.load_state_dict(joint.state_dict())
    joint_output, joint_diagnostics = joint(x, return_diagnostics=True)
    shuffled_output, shuffled_diagnostics = shuffled(x, return_diagnostics=True)
    assert_close(joint_diagnostics.attention, shuffled_diagnostics.attention, atol=0.0)
    if torch.allclose(joint_output, shuffled_output):
        raise AssertionError("Shuffled pairing did not change the relation output.")

    uniform_attention = diagnostics["uniform_global"].attention
    for row in uniform_attention.reshape(-1, uniform_attention.shape[-1]):
        allowed_values = row[row > 0]
        if allowed_values.numel() == 0:
            raise AssertionError("Uniform-global attention has an empty row.")
        if not torch.allclose(
            allowed_values,
            allowed_values[0].expand_as(allowed_values),
            atol=1e-7,
            rtol=0.0,
        ):
            raise AssertionError("Uniform-global attention is not uniform on allowed pairs.")


def test_width_matched_paper_unet() -> None:
    model = PaperWidthDualUNet1D(in_channels=1, base_width=20).eval()
    parameter_count = count_trainable_parameters(model)
    if parameter_count != 100642:
        raise AssertionError(f"Unexpected width-matched parameter count: {parameter_count}.")
    with torch.no_grad():
        output = model(torch.randn(1, 1, 451))
    if output.shape != (1, 2, 451):
        raise AssertionError(f"Unexpected width-matched output shape: {tuple(output.shape)}")


def test_cuda_parity_if_available() -> str:
    if not torch.cuda.is_available():
        return "SKIP (CUDA unavailable)"
    torch.manual_seed(19)
    cpu_layer = RelationConv1D(
        2,
        8,
        relation_dim=8,
        num_heads=2,
        degree=3,
        gate_init=0.2,
    ).eval()
    gpu_layer = RelationConv1D(
        2,
        8,
        relation_dim=8,
        num_heads=2,
        degree=3,
        gate_init=0.2,
    ).cuda().eval()
    gpu_layer.load_state_dict(cpu_layer.state_dict())
    x = torch.randn(2, 2, 31)
    with torch.no_grad():
        cpu_output = cpu_layer(x)
        gpu_output = gpu_layer(x.cuda()).cpu()
    assert_close(gpu_output, cpu_output, atol=2e-5)
    return torch.cuda.get_device_name(0)


def test_full_dual_unet_adam_step_if_cuda_available() -> str:
    if not torch.cuda.is_available():
        return "SKIP (CUDA unavailable)"
    torch.manual_seed(20)
    model = PaperRelationDualUNet1D(
        in_channels=1,
        relation_dim=8,
        num_heads=2,
        degree=3,
        nonlocal_radius=2,
        gate_init=0.1,
    ).cuda().train()
    relation_layers = [
        module for module in model.modules() if isinstance(module, RelationConv1D)
    ]
    if len(relation_layers) != 10:
        raise AssertionError(f"Expected 10 relation layers, found {len(relation_layers)}.")
    first_gate_before = relation_layers[0].gate.detach().clone()
    first_coefficients_before = relation_layers[0].relation.coefficients.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(1, 1, 451, device="cuda")
    target = torch.randn(1, 2, 451, device="cuda")
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.l1_loss(model(x), target)
    loss.backward()
    for index, layer in enumerate(relation_layers):
        if layer.gate.grad is None or not torch.isfinite(layer.gate.grad):
            raise AssertionError(f"Invalid gate gradient in relation layer {index}.")
        coefficient_gradient = layer.relation.coefficients.grad
        if coefficient_gradient is None or not torch.isfinite(coefficient_gradient).all():
            raise AssertionError(f"Invalid coefficient gradient in relation layer {index}.")
        active = coefficient_gradient[
            layer.relation.pair_mask[None].expand_as(coefficient_gradient)
        ]
        if float(active.abs().sum()) <= 0.0:
            raise AssertionError(f"No active coefficient gradient in relation layer {index}.")
    optimizer.step()
    if torch.equal(relation_layers[0].gate.detach(), first_gate_before):
        raise AssertionError("Adam did not update the relation gate.")
    active_mask = relation_layers[0].relation.pair_mask[None].expand_as(
        first_coefficients_before
    )
    coefficient_change = (
        relation_layers[0].relation.coefficients.detach() - first_coefficients_before
    )[active_mask]
    if float(coefficient_change.abs().sum()) <= 0.0:
        raise AssertionError("Adam did not update active polynomial coefficients.")
    return f"loss={float(loss.detach()):.6f}, layers={len(relation_layers)}"


def main() -> None:
    torch.set_num_threads(1)
    tests = [
        test_chebyshev_values,
        test_attention_and_mask,
        test_factorized_relation_matches_explicit_pairs,
        test_zero_gate_is_exact_local_convolution,
        test_zero_gate_can_wake_up,
        test_relation_parameters_receive_gradients_when_gate_opens,
        test_gate_gradient_matches_finite_difference,
        test_relation_layer_shapes_for_all_unet_sites,
        test_full_dual_unet_nests_original_at_zero_gate,
        test_seed_matched_backbone_initialization,
        test_state_dict_round_trip,
        test_relation_ablation_modes,
        test_width_matched_paper_unet,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    cuda_status = test_cuda_parity_if_available()
    print(f"PASS test_cuda_parity_if_available: {cuda_status}", flush=True)
    adam_status = test_full_dual_unet_adam_step_if_cuda_available()
    print(f"PASS test_full_dual_unet_adam_step_if_cuda_available: {adam_status}", flush=True)
    print(f"ALL_TESTS_PASSED count={len(tests) + 2}", flush=True)


if __name__ == "__main__":
    main()
