import torch
import numpy as np
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.
    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div,
                torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div,
                    torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta


def optimize_linear(grad, eps, norm=np.inf):
    """
    Solves for the optimal input to a linear function under a norm constraint.
    Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
    :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
    :param eps: float. Scalar specifying size of constraint region
    :param norm: np.inf, 1, or 2. Order of norm constraint.
    :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
    """

    red_ind = list(range(1, len(grad.size())))
    avoid_zero_div = torch.tensor(1e-12, dtype=grad.dtype, device=grad.device)
    if norm == np.inf:
        # Take sign of gradient
        optimal_perturbation = torch.sign(grad)
    elif norm == 1:
        abs_grad = torch.abs(grad)
        sign = torch.sign(grad)
        red_ind = list(range(1, len(grad.size())))
        abs_grad = torch.abs(grad)
        ori_shape = [1] * len(grad.size())
        ori_shape[0] = grad.size(0)

        max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
        max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
        num_ties = max_mask
        for red_scalar in red_ind:
            num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
        optimal_perturbation = sign * max_mask / num_ties
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
        assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
    elif norm == 2:
        square = torch.max(
            avoid_zero_div,
            torch.sum(grad ** 2, red_ind, keepdim=True)
        )
        optimal_perturbation = grad / torch.sqrt(square)
        # TODO integrate below to a test file
        # check that the optimal perturbations have been correctly computed
        opt_pert_norm = (
            optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
        )
        one_mask = (square <= avoid_zero_div).to(
            torch.float) * opt_pert_norm + (
                           square > avoid_zero_div
                   ).to(torch.float)
        assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
    else:
        raise NotImplementedError(
            "Only L-inf, L1 and L2 norms are " "currently implemented."
        )

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = eps * optimal_perturbation
    return scaled_perturbation


def fast_gradient_method(
        model_fn,
        x,
        eps,
        norm,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        sanity_checks=False,
):
    """
    PyTorch implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError(
            "Norm order must be either np.inf, 1, or 2, got {} instead.".format(
                norm)
        )
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # x needs to be a leaf variable, of floating point type and have requires_grad being True for
    # its grad to be computed and stored properly in a backward call
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x)[0], 1)

    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(model_fn(x)[0], y)
    # If attack is targeted, minimize loss of target label rather than maximize loss of correct label
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    loss.backward()
    optimal_perturbation = optimize_linear(x.grad, eps, norm)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
            )
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def basic_iterative_method(
        model,
        x,
        y=None,
        targeted=False,
        eps=0.15,
        eps_iter=0.01,
        n_iter=50,
        clip_max=None,
        clip_min=None):
    x = x.clone().detach().to(torch.float).requires_grad_(True)
    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model(x)[0], 1)

    eta = torch.zeros(x.shape).to(x.device)

    for i in range(n_iter):
        out = model(x + eta)[0]
        loss = F.cross_entropy(out, y)
        if targeted:
            loss = -loss
        loss.backward()
        eta += eps_iter * torch.sign(x.grad.data)
        eta.clamp_(-eps, eps)
        x.grad.data.zero_()

    x_adv = x + eta

    if (clip_min is not None) or (clip_max is not None):
        if clip_min is None or clip_max is None:
            raise ValueError(
                "One of clip_min and clip_max is None but we don't currently support one-sided clipping")
        x_adv.clamp_(clip_min, clip_max)

    return x_adv.detach()


INF = float("inf")


def carlini_wagner_l2(
        model, imgs, targets,
        batch_size=1000, device='cuda', **kwargs):
    """
    The L_2 optimized attack.

    This attack is the most efficient and should be used as the primary
    attack to evaluate potential defenses.

    Args:
        model: A callable that takes an input tensor and returns the model
          logits.
        imgs: Input tensor.
        targets: If targeted is true, then the targets represents the target
          labels. Otherwise,then targets are the original class labels.
        batch_size: Number of attacks to run simultaneously.
        device: A device to run the attack.

    Returns:
        Adversarial examples for the supplied model.
    """

    imgs = imgs.to(device)
    targets = targets.to(device)
    layers = [module for module in model.modules()]
    output_layer = layers[-1]
    num_classes = output_layer.out_features
    targets = F.one_hot(targets, num_classes=num_classes)
    model = model.to(device)

    adv_imgs = []
    kwargs['device'] = device
    for i in range(0, len(imgs), batch_size):
        img_batch = imgs[i:i + batch_size]
        target_batch = targets[i:i + batch_size]
        attack_result = cw_l2_attack_batch(model, img_batch, target_batch,
                                           **kwargs)
        attack_result = torch.stack(attack_result)
        adv_imgs += [attack_result]

    adv_imgs = torch.cat(adv_imgs)
    return adv_imgs


def cw_l2_attack_batch(model, imgs, targets,
                       confidence=0, targeted=False, learning_rate=1e-2,
                       binary_search_steps=9, max_iterations=10000,
                       abort_early=True,
                       initial_const=1e-3,
                       boxmin=-0.5, boxmax=0.5, device='cuda'):
    """
    Performs binary search to find optimal constant value.

    Args:
        model: A callable that takes an input tensor and returns the model
          logits.
        imgs: Input tensor.
        targets: If targeted is true, then the targets represents the target
          labels. Otherwise,then targets are the original class labels.
        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller
          values produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and
          will produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets
          stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is
          large, the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        device: A device to run the attack.

    Returns:
        Adversarial examples for the supplied model.
    """

    def arctanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def compare(x, y):
        if isinstance(x, torch.Tensor) and x.dtype != torch.int64:
            x = x.clone().detach()
            if targeted:
                x[y] -= confidence
            else:
                x[y] += confidence
            x = torch.argmax(x)
        if targeted:
            return x == y
        else:
            return x != y

    def convert_from_tanh(imgs):
        # The resulting image, tanh'd to keep bounded from boxmin to boxmax
        boxmul = (boxmax - boxmin) / 2.
        boxplus = (boxmin + boxmax) / 2.
        return torch.tanh(imgs) * boxmul + boxplus

    def convert2tanh(imgs):
        boxmul = (boxmax - boxmin) / 2.
        boxplus = (boxmin + boxmax) / 2.
        return arctanh((imgs - boxplus) / boxmul * 0.999999)

    def get_target_logit(targets, logits):
        return torch.sum(targets * logits, dim=1)

    def get_others_max_logit(targets, logits):
        logits = (1 - targets) * logits - (targets  * 10000)
        return torch.max(logits, dim=1)[0]

    def compute_loss(tanh_imgs, targets, modifier, const):
        tanh_adv_imgs = tanh_imgs + modifier
        adv_imgs = convert_from_tanh(tanh_adv_imgs)
        imgs = convert_from_tanh(tanh_imgs)

        logit_adv_img = model(adv_imgs)[0]
        real = get_target_logit(targets, logit_adv_img)
        other = get_others_max_logit(targets, logit_adv_img)

        if targeted:
            # If targeted, optimize for making the other class most likely.
            loss1 = torch.max(torch.tensor([0.0]).to(device),
                              other - real + confidence)
        else:
            # If untargeted, optimize for making this class least likely.
            loss1 = torch.max(torch.tensor([0.0]).to(device),
                              real - other + confidence)

        diff = adv_imgs - imgs
        l2dist_loss = torch.sum(diff * diff, dim=[1,2,3])

        loss1 = torch.sum(const * loss1)
        loss2 = torch.sum(l2dist_loss)
        loss = loss1 + loss2

        return loss, l2dist_loss, logit_adv_img, adv_imgs

    # Set the lower and upper bounds accordingly
    batch_size = imgs.shape[0]
    lower_bound = torch.zeros(batch_size).to(device)
    const = torch.ones(batch_size).to(device) * initial_const
    upper_bound = torch.ones(batch_size).to(device) * 1e10

    # The best l2, score, and image attack
    o_bestl2 = [1e10] * batch_size
    o_bestscore = [-1] * batch_size
    o_bestattack = [torch.zeros(imgs[0].shape).to(device)] * batch_size

    tanh_imgs = convert2tanh(imgs)

    for outer_step in range(binary_search_steps):
        print(('[%d/%d] CW L2 attack binary search step' %
               (outer_step + 1, binary_search_steps)))

        # Completely reset internal state.
        modifier = torch.zeros_like(imgs, requires_grad=True)
        modifier = modifier.to(device)
        optimizer = optim.Adam([modifier], lr=learning_rate)

        bestl2 = [1e10] * batch_size
        bestscore = [-1] * batch_size

        # The last iteration (if we run many steps) repeat the search once.
        repeat = binary_search_steps >= 10
        if repeat == True and outer_step == binary_search_steps - 1:
            const = upper_bound

        prev = np.inf

        for iteration in tqdm(range(max_iterations)):
            # Perform the attack
            (loss, l2_dists,
            logits, adv_imgs) = compute_loss(tanh_imgs, targets,
                                             modifier, const)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if we should abort search if we're getting nowhere.
            if abort_early and iteration % (max_iterations // 10) == 0:
                if loss > prev * .9999: break
                prev = loss

            # Adjust the best result found so far
            zipped_result = zip(l2_dists, logits, adv_imgs)
            for idx, (l2_dist, logit, adv_img) in enumerate(zipped_result):
                if (l2_dist < bestl2[idx] and
                    compare(logit, torch.argmax(targets[idx]))):
                    bestl2[idx] = l2_dist
                    bestscore[idx] = torch.argmax(logit)

                if (l2_dist < o_bestl2[idx] and
                    compare(logit, torch.argmax(targets[idx]))):
                    o_bestl2[idx] = l2_dist
                    o_bestscore[idx] = torch.argmax(logit)
                    o_bestattack[idx] = adv_img

        # Adjust the constant as needed
        for idx in range(batch_size):
            if (compare(bestscore[idx], torch.argmax(targets[idx])) and
                bestscore[idx] != -1):
                # On success, divide const by two
                upper_bound[idx] = min(upper_bound[idx], const[idx])
                if upper_bound[idx] < 1e9:
                    const[idx] = (lower_bound[idx] + upper_bound[idx]) / 2
            else:
                # On failure, either multiply by 10 if no solution found yet
                # or do binary search with the known upper bound
                lower_bound[idx] = max(lower_bound[idx], const[idx])
                if upper_bound[idx] < 1e9:
                    const[idx] = (lower_bound[idx] + upper_bound[idx])/2
                else:
                    const[idx] *= 10

    # Return the best solution found
    return o_bestattack


# def carlini_wagner_l2(
#         model,
#         images,
#         labels,
#         targeted=False, c=1, kappa=0,
#         max_iter=1000, learning_rate=5e-3, device="cpu"):
#     images = images.to(device)
#     labels = labels.to(device)
#
#     # Define f-function
#     def f(x):
#         outputs = model(x)[0]
#         one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
#
#         i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
#
#         j = torch.masked_select(outputs, one_hot_labels.bool())
#
#         # If targeted, optimize for making the other class most likely
#         if targeted:
#             return torch.clamp(i - j, min=-kappa)
#
#         # If untargeted, optimize for making the other class most likely
#         else:
#             return torch.clamp(j - i, min=-kappa)
#
#     w = torch.zeros_like(images, requires_grad=True).to(device)
#
#     optimizer = optim.Adam([w], lr=learning_rate)
#
#     prev = 1e10
#
#     for step in tqdm(range(max_iter), leave=False):
#
#         a = 1 / 2 * (nn.Tanh()(w) + 1)
#
#         loss1 = nn.MSELoss(reduction='sum')(a, images)
#         loss2 = torch.sum(c * f(a))
#
#         cost = loss1 + loss2
#
#         optimizer.zero_grad()
#         cost.backward()
#         optimizer.step()
#
#         # Early Stop when loss does not converge.
#         if step % (max_iter // 10) == 0:
#             if cost > prev:
#                 return a
#             prev = cost
#
#     attack_images = 1 / 2 * (nn.Tanh()(w) + 1)
#
#     return attack_images


def projected_gradient_descent(
        model_fn,
        x,
        eps,
        eps_iter,
        nb_iter,
        norm,
        clip_min=None,
        clip_max=None,
        y=None,
        targeted=False,
        rand_init=True,
        rand_minmax=None,
        sanity_checks=True,
):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to False. or the
    Madry et al. (2017) method if rand_init is set to True.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
    :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
              which the random perturbation on x was drawn. Effective only when rand_init is
              True. Default equals to eps.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")
    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x
    if eps_iter < 0:
        raise ValueError(
            "eps_iter must be greater than or equal to 0, got {} instead".format(
                eps_iter
            )
        )
    if eps_iter == 0:
        return x

    assert eps_iter <= eps, (eps_iter, eps)
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        assert_ge = torch.all(
            torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_ge)

    if clip_max is not None:
        assert_le = torch.all(
            torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype))
        )
        asserts.append(assert_le)

    # Initialize loop variables
    if rand_init:
        if rand_minmax is None:
            rand_minmax = eps
        eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
    else:
        eta = torch.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = torch.clamp(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        _, y = torch.max(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = torch.clamp(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast clip_min and clip_max to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def spsa(
        model_fn,
        x,
        eps,
        nb_iter,
        norm=np.inf,
        clip_min=-np.inf,
        clip_max=np.inf,
        y=None,
        targeted=False,
        early_stop_loss_threshold=None,
        learning_rate=0.01,
        delta=0.01,
        spsa_samples=128,
        spsa_iters=1,
        is_debug=False,
        sanity_checks=True,
):
    """
    This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666
    (Uesato et al. 2018). SPSA is a gradient-free optimization method, which is useful when
    the model is non-differentiable, or more generally, the gradients do not point in useful
    directions.
    :param model_fn: A callable that takes an input tensor and returns the model logits.
    :param x: Input tensor.
    :param eps: The size of the maximum perturbation, measured in the L-infinity norm.
    :param nb_iter: The number of optimization steps.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: If specified, the minimum input value.
    :param clip_max: If specified, the maximum input value.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted? Untargeted, the
              default, will try to make the label incorrect. Targeted will instead try to
              move in the direction of being more like y.
    :param early_stop_loss_threshold: A float or None. If specified, the attack will end as
              soon as the loss is below `early_stop_loss_threshold`.
    :param learning_rate: Learning rate of ADAM optimizer.
    :param delta: Perturbation size used for SPSA approximation.
    :param spsa_samples:  Number of inputs to evaluate at a single time. The true batch size
              (the number of evaluated inputs for each update) is `spsa_samples *
              spsa_iters`
    :param spsa_iters:  Number of model evaluations before performing an update, where each
              evaluation is on `spsa_samples` different inputs.
    :param is_debug: If True, print the adversarial loss after each update.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """

    if y is not None and len(x) != len(y):
        raise ValueError(
            "number of inputs {} is different from number of labels {}".format(
                len(x), len(y)
            )
        )
    if y is None:
        y = torch.argmax(model_fn(x)[0], dim=1)

    # The rest of the function doesn't support batches of size greater than 1,
    # so if the batch is bigger we split it up.
    if len(x) != 1:
        adv_x = []
        for x_single, y_single in tqdm(zip(x, y), leave=False, total=len(x)):
            adv_x_single = spsa(
                model_fn=model_fn,
                x=x_single.unsqueeze(0),
                eps=eps,
                nb_iter=nb_iter,
                norm=norm,
                clip_min=clip_min,
                clip_max=clip_max,
                y=y_single.unsqueeze(0),
                targeted=targeted,
                early_stop_loss_threshold=early_stop_loss_threshold,
                learning_rate=learning_rate,
                delta=delta,
                spsa_samples=spsa_samples,
                spsa_iters=spsa_iters,
                is_debug=is_debug,
                sanity_checks=sanity_checks,
            )
            adv_x.append(adv_x_single)
        return torch.cat(adv_x)

    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x

    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    asserts.append(torch.all(x >= clip_min))
    asserts.append(torch.all(x <= clip_max))

    if is_debug:
        print("Starting SPSA attack with eps = {}".format(eps))

    perturbation = (torch.rand_like(x) * 2 - 1) * eps
    _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)
    optimizer = optim.Adam([perturbation], lr=learning_rate)

    for i in range(nb_iter):

        def loss_fn(pert):
            """
            Margin logit loss, with correct sign for targeted vs untargeted loss.
            """
            logits = model_fn(x + pert)[0]
            loss_multiplier = 1 if targeted else -1
            return loss_multiplier * _margin_logit_loss(logits,
                                                        y.expand(len(pert)))

        spsa_grad = _compute_spsa_gradient(
            loss_fn, x, delta=delta, samples=spsa_samples, iters=spsa_iters
        )
        perturbation.grad = spsa_grad
        optimizer.step()

        _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)

        loss = loss_fn(perturbation).item()
        if is_debug:
            print("Iteration {}: loss = {}".format(i, loss))
        if early_stop_loss_threshold is not None and loss < early_stop_loss_threshold:
            break

    adv_x = torch.clamp((x + perturbation).detach(), clip_min, clip_max)

    if norm == np.inf:
        asserts.append(torch.all(torch.abs(adv_x - x) <= eps + 1e-6))
    else:
        asserts.append(
            torch.all(
                torch.abs(
                    torch.renorm(adv_x - x, p=norm, dim=0, maxnorm=eps) - (
                            adv_x - x)
                )
                < 1e-6
            )
        )
    asserts.append(torch.all(adv_x >= clip_min))
    asserts.append(torch.all(adv_x <= clip_max))

    if sanity_checks:
        assert np.all(asserts)

    return adv_x


def _project_perturbation(
        perturbation, norm, epsilon, input_image, clip_min=-np.inf,
        clip_max=np.inf
):
    """
    Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into
    hypercube such that the resulting adversarial example is between clip_min and clip_max,
    if applicable. This is an in-place operation.
    """

    clipped_perturbation = clip_eta(perturbation, norm, epsilon)
    new_image = torch.clamp(input_image + clipped_perturbation, clip_min,
                            clip_max)

    perturbation.add_((new_image - input_image) - perturbation)


def _compute_spsa_gradient(loss_fn, x, delta, samples, iters):
    """
    Approximately compute the gradient of `loss_fn` at `x` using SPSA with the
    given parameters. The gradient is approximated by evaluating `iters` batches
    of `samples` size each.
    """

    assert len(x) == 1
    num_dims = len(x.size())

    x_batch = x.expand(samples, *([-1] * (num_dims - 1)))

    grad_list = []
    for i in range(iters):
        delta_x = delta * torch.sign(torch.rand_like(x_batch) - 0.5)
        delta_x = torch.cat([delta_x, -delta_x])
        with torch.no_grad():
            loss_vals = loss_fn(x + delta_x)
        while len(loss_vals.size()) < num_dims:
            loss_vals = loss_vals.unsqueeze(-1)
        avg_grad = (
                torch.mean(loss_vals * torch.sign(delta_x), dim=0,
                           keepdim=True) / delta
        )
        grad_list.append(avg_grad)

    return torch.mean(torch.cat(grad_list), dim=0, keepdim=True)


def _margin_logit_loss(logits, labels):
    """
    Computes difference between logits for `labels` and next highest logits.
    The loss is high when `label` is unlikely (targeted by default).
    """

    correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

    logit_indices = torch.arange(
        logits.size()[1],
        dtype=labels.dtype,
        device=labels.device,
    )[None, :].expand(labels.size()[0], -1)
    incorrect_logits = torch.where(
        logit_indices == labels[:, None],
        torch.full_like(logits, float("-inf")),
        logits,
    )
    max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

    return max_incorrect_logits - correct_logits
