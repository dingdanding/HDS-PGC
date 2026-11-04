"""Asymmetric, group-wise auxiliary gradient projection for DSFN-J.

PCGrad-inspired: preserve RGB+mesh gradients and remove only the opposing
component of the depth gradient. Not symmetric multitask PCGrad, not an Adam
descent guarantee. No extra forward or inference branch is introduced.
"""
import math
import torch

GROUPS = ('backbone', 'global_head', 'local_head')


class DepthGradientProjector:
    def __init__(self, model, eps=1e-12):
        self.eps = eps
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = tuple(p for _, p in self.named)
        self.groups = {g: [] for g in GROUPS}
        for i, (name, _) in enumerate(self.named):
            if name.startswith('feature_extractor'):
                group = 'backbone'
            elif name.startswith('regressNet1'):
                group = 'global_head'
            elif name.startswith('regressNet2'):
                group = 'local_head'
            else:
                raise ValueError('Unassigned trainable parameter: ' + name)
            self.groups[group].append(i)
        if not all(self.groups.values()):
            raise ValueError('Expected backbone, global and local DSFN parameter groups')

    def backward(self, main_loss, auxiliary_loss, enabled=True):
        """Caller must zero gradients before this call and clip/step afterwards.

        Store only auxiliary gradients separately. main_loss.backward() writes
        directly into parameter.grad and releases the graph before projection.
        Thus we avoid retaining two separate full gradient lists PLUS .grad.
        """
        if any(p.grad is not None for p in self.params):
            raise RuntimeError('Call optimizer.zero_grad(set_to_none=True) first')
        if not bool(torch.isfinite(main_loss).item() and torch.isfinite(auxiliary_loss).item()):
            raise FloatingPointError('Non-finite main/depth loss before projection')
        if not enabled:
            (main_loss + auxiliary_loss).backward()
            return {}

        auxiliary = torch.autograd.grad(auxiliary_loss, self.params,
                                        retain_graph=True, allow_unused=True)
        main_loss.backward()
        stats = {}
        with torch.no_grad():
            for group, indices in self.groups.items():
                # Include zero/unused components in each group's Euclidean
                # geometry, but preserve None for parameters unused by BOTH.
                device = self.params[indices[0]].device
                dot = torch.zeros((), device=device)
                main_sq = torch.zeros((), device=device)
                aux_sq = torch.zeros((), device=device)
                for i in indices:
                    a, b = self.params[i].grad, auxiliary[i]
                    if a is not None:
                        main_sq += torch.dot(a.flatten(), a.flatten())
                    if b is not None:
                        aux_sq += torch.dot(b.flatten(), b.flatten())
                    if a is not None and b is not None:
                        dot += torch.dot(a.flatten(), b.flatten())
                d, pp, dd = (float(v.item()) for v in (dot, main_sq, aux_sq))
                if not all(math.isfinite(v) for v in (d, pp, dd)):
                    raise FloatingPointError('Non-finite gradients in ' + group)
                project = d < 0.0 and pp > self.eps
                coefficient = d / pp if project else 0.0
                for i in indices:
                    parameter, b = self.params[i], auxiliary[i]
                    a = parameter.grad
                    if a is None:
                        if b is not None:
                            parameter.grad = b
                    else:
                        if project:
                            a.mul_(1.0 - coefficient)
                        if b is not None:
                            a.add_(b)
                projected_sq = max(0.0, dd - 2*coefficient*d + coefficient*coefficient*pp)
                stats[group] = {
                    'conflict': float(project),
                    'cosine': d / max(math.sqrt(pp*dd), self.eps),
                    'depth_main_norm_ratio': math.sqrt(dd / max(pp, self.eps)),
                    'aux_main_dot_after': d - coefficient*pp,
                    'depth_norm_after': math.sqrt(projected_sq),
                    'small_main_norm': float(pp <= self.eps),
                }
        return stats
