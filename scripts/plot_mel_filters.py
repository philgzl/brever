import matplotlib.pyplot as plt
import torch

from brever.modules import MelFilterbank


def main():
    mel_fb = MelFilterbank(64)

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(mel_fb.filters.T)
    axes[0].set_title('filters')
    axes[1].plot(mel_fb.inverse_filters.T)
    axes[1].set_title('inverse filters')
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(mel_fb.inverse_filters@mel_fb.filters)
    plt.colorbar(im, ax=ax)
    ax.set_title('analysis-synthesis function')
    fig.tight_layout()

    def plot(ax, data, title, vmin, vmax):
        im = ax.imshow(data, aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)

    def example(x):
        fig, axes = plt.subplots(3, 1)
        y = mel_fb.filters@x
        z = mel_fb.inverse_filters@y
        vmin = min(x.min(), y.min(), z.min())
        vmax = max(x.max(), y.max(), z.max())
        plot(axes[0], x, 'input', vmin, vmax)
        plot(axes[1], y, 'analysis', vmin, vmax)
        plot(axes[2], z, 'synthesis', vmin, vmax)
        fig.tight_layout()

    example(torch.rand(257, 500))
    example(torch.randn(257, 500))
    example(torch.ones(257, 500))

    plt.show()


if __name__ == '__main__':
    main()
