import os

import torch

from brever.modules import EMAKarras

if __name__ == '__main__':
    import tempfile

    import matplotlib.pyplot as plt

    n_updates = 100
    n_checkpoints = 10

    signal = torch.sin(torch.linspace(0, torch.pi, n_updates)) \
        + 0.1 * torch.randn(n_updates)
    t_i = torch.linspace(0, n_updates - 1, n_checkpoints + 1)
    t_i = t_i[1:].long().tolist()
    t_r = torch.arange(1, n_updates).tolist()

    sigma_rel_i = [0.05, 0.10]
    sigma_rel_r = 0.20

    model = torch.nn.Linear(1, 1, bias=False)
    ema = EMAKarras(model, sigma_rels=sigma_rel_i + [sigma_rel_r])

    averaged_signals = []

    with tempfile.TemporaryDirectory() as tempdir:
        for i, val in enumerate(signal):
            with torch.no_grad():
                model.weight.fill_(val)
            ema.update()
            averaged_signals.append([
                ema.ema_params[sigma_rel][0].item()
                for sigma_rel in ema.sigma_rels
            ])
            if i in t_i:
                torch.save(
                    ema.state_dict(), os.path.join(tempdir, f'{i}.ckpt')
                )

        ema_2 = EMAKarras(model, sigma_rels=sigma_rel_i)
        post_hoc = ema_2.post_hoc_ema(tempdir, sigma_rel_r, t_r=t_r,
                                      apply=False)
        post_hoc = torch.tensor(post_hoc).squeeze()

        fig, ax = plt.subplots()
        ax.plot(signal, label='signal')
        for sigma_rel, avg_signal in zip(
            ema.sigma_rels, zip(*averaged_signals)
        ):
            label = f'sigma_rel={sigma_rel}'
            if sigma_rel == sigma_rel_r:
                label += ' (true)'
            ax.plot(avg_signal, label=label)

        ax.plot(t_r, post_hoc, label=f'sigma_rel={sigma_rel_r} (post-hoc)',
                linestyle='--')

        ax.plot(t_i, signal[t_i], linestyle='none', marker='x',
                label='checkpoints', c='k')

        ax.legend()
        plt.show()
